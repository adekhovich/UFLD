"""UFLD: full single-model pipeline.

Stages (run in this order, any subset can be selected with --stages):

  pretrain  train the causal Transformer classifier         (notebooks/0-pretrain.ipynb)
  rl        train the PPO patch selector on top of it        (notebooks/1-rl.ipynb)
  cp        conformal A/B calibration + test evaluation      (notebooks/2-CP_inference_cls.ipynb)

A stage that is not selected is not re-run: its checkpoint is loaded from --models-dir.

Examples
--------
  # everything, large model, seed 4
  python main.py --seed 4 --model-size large

  # only calibration/evaluation from existing checkpoints, with ResNet-34 fallback and the alpha sweep
  python main.py --stages cp --hybrid --alpha-sweep
"""
import argparse
import json
import os
import pickle


STAGES = ("pretrain", "rl", "cp")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="UFLD single-model pipeline",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("general")
    g.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES), help="stages to run")
    g.add_argument("--seed", type=int, default=4)
    g.add_argument("--model-size", choices=("small", "medium", "large"), default="large",
                   help="Transformer configuration (Table 1)")
    g.add_argument("--data-path", default="data/MixedWM38.npz")
    g.add_argument("--models-dir", default="models")
    g.add_argument("--results-dir", default="results")
    g.add_argument("--batch-size", type=int, default=128)
    g.add_argument("--device", default=None, help="e.g. cuda:0 or cpu (default: cuda:0 if available)")

    g = p.add_argument_group("pretrain")
    g.add_argument("--cls-epochs", type=int, default=240)
    g.add_argument("--cls-lr", type=float, default=1e-3)
    g.add_argument("--cls-wd", type=float, default=1e-6)
    g.add_argument("--supcon-weight", type=float, default=0.1, help="weight c of the supervised contrastive loss")
    g.add_argument("--num-output-heads", type=int, default=64,
                   help="random prefix positions supervised per batch")

    g = p.add_argument_group("rl")
    g.add_argument("--rl-epochs", type=int, default=100)
    g.add_argument("--rl-lr", type=float, default=1e-3)
    g.add_argument("--gamma", type=float, default=0.99)
    g.add_argument("--gae-lambda", type=float, default=0.99)
    g.add_argument("--policy-clip", type=float, default=0.2)

    g = p.add_argument_group("cp")
    g.add_argument("--alpha", type=float, default=0.065, help="miscoverage level")
    g.add_argument("--hybrid", action="store_true",
                   help="also evaluate with ResNet-34 fallback when the stopping rule never fires")
    g.add_argument("--cnn-checkpoint", default=None,
                   help="ResNet-34 checkpoint (default: <models-dir>/mixedwm38/resnet34_seed<seed>.pth)")
    g.add_argument("--alpha-sweep", action="store_true",
                   help="also evaluate alpha in [0.05, 0.10] with step 0.005 (Table 5)")
    g.add_argument("--m", type=int, default=4, help="maximum number of predicted positive labels at stopping")
    g.add_argument("--min-steps", type=int, default=1)
    g.add_argument("--patience", type=int, default=1)

    return p.parse_args(argv)


def run(args):
    # Heavy imports here so that `python main.py --help` works without the full environment
    import numpy as np
    import torch
    from sklearn.metrics import f1_score

    from src.config import MODEL_CONFIGS, classifier_checkpoint, ppo_model_name
    from src.models.classifier.resnet import init_resnet
    from src.models.classifier.transformer import CausalTransformerSupCon
    from src.models.patch_selection.classifier import CausalTransformerClassifierKVCached
    from src.models.patch_selection.environment import WaferMapGrid
    from src.models.patch_selection.ppo import PPOAgent
    from src.trainer.train_ppo import training_loop_ppo_transformer
    from src.trainer.train_transformer import training_loop_transformer
    from src.utils.conformal import (
        make_AB_loaders,
        calibrate_multilabel_CP_two_rule_AB,
        evaluate_multilabel_CP_two_rule,
    )
    from src.utils.contrastive_loss import SupConLoss
    from src.utils.dataloader import load_mixedwm38_splits, get_loaders
    from src.utils.utils import seed_everything, accuracy_transformer

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Fixed by the dataset and the paper's setup
    dataset_name = "mixedwm38"
    num_classes = 8
    input_channels = 1
    img_size = (52, 52)
    patch_size = 4
    num_patches = (img_size[0] // patch_size) ** 2   # 169
    num_heads = 4
    reward_fun = "sparse"

    embedding_dim, num_layers = MODEL_CONFIGS[args.model_size]
    hidden_dim = 2 * embedding_dim

    cls_path = classifier_checkpoint(dataset_name, args.seed, num_layers, hidden_dim, num_heads=num_heads,
                                     num_output_heads=args.num_output_heads, c=args.supcon_weight,
                                     models_dir=args.models_dir)
    ppo_name = ppo_model_name(dataset_name, num_layers, hidden_dim, num_heads=num_heads,
                              num_output_heads=args.num_output_heads, c=args.supcon_weight,
                              reward_fun=reward_fun)
    tag = f"{args.model_size}_seed{args.seed}"

    def data_loaders():
        """D1 (train), D3 (val, patch-selector training), D2 (calib), D4 (test)."""
        train_ds, val_ds, calib_ds, test_ds = load_mixedwm38_splits(
            args.data_path, img_size=img_size, input_channels=input_channels, num_classes=num_classes)
        train_loader, test_loader, val_loader, _ = get_loaders(
            train_ds, test_ds, val_dataset=val_ds, calib_dataset=calib_ds, batch_size=args.batch_size)
        return train_loader, val_loader, test_loader, calib_ds

    def frozen_classifier():
        """Pretrained Transformer with KV cache for step-by-step acquisition."""
        clf = CausalTransformerClassifierKVCached(
            input_dim=patch_size ** 2, hidden_dim=hidden_dim, num_classes=num_classes, seq_len=num_patches,
            num_layers=num_layers, num_heads=num_heads, patch_size=patch_size, dropout_prob=0.1,
            batch_first=True, layer_norm=True, use_cls_token=False, embedding_dim=embedding_dim,
            use_patch_id_emb=True,
        )
        clf.load_weights(cls_path)
        clf = clf.to(device)
        clf.eval()
        clf.freeze_all()
        return clf

    def ppo_agent(env, clf):
        return PPOAgent(n_actions=env.action_space.n, obs_dim=clf.embed_dim, batch_size=args.batch_size,
                        gamma=args.gamma, alpha=args.rl_lr, gae_lambda=args.gae_lambda, n_epochs=args.rl_epochs,
                        policy_clip=args.policy_clip, seed=args.seed, chkpt_dir=args.models_dir,
                        model_name=ppo_name, device=device)

    # ------------------------------------------------------------------ stage 1: pretrain
    if "pretrain" in args.stages:
        print(f"\n=== [pretrain] causal Transformer ({args.model_size}), seed {args.seed} ===")
        seed_everything(seed=args.seed)

        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)

        classifier = CausalTransformerSupCon(
            input_dim=patch_size ** 2, hidden_dim=hidden_dim, num_classes=num_classes, seq_len=num_patches,
            num_layers=num_layers, num_heads=num_heads, patch_size=patch_size, dropout_prob=0.1,
            dataset_name=dataset_name, batch_first=True, layer_norm=True, use_cls_token=False,
            embedding_dim=embedding_dim, permute_patches=True, use_patch_id_emb=True,
        ).to(device)

        train_loader, val_loader, test_loader, _ = data_loaders()

        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.cls_lr, weight_decay=args.cls_wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cls_epochs, eta_min=1e-6)

        training_loop_transformer(
            model=classifier,
            criterion=torch.nn.BCEWithLogitsLoss(reduction="none"),
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            valid_loader=val_loader,
            epochs=args.cls_epochs,
            num_output_heads=args.num_output_heads,
            supcon_loss=SupConLoss(),
            supcon_weight=args.supcon_weight,
            file_name=cls_path,
            device=device,
        )

        # best checkpoint (by validation accuracy) on the full sequence of patches
        classifier.load_state_dict(torch.load(cls_path, map_location=device))
        seed_everything(seed=args.seed)
        acc, f1 = accuracy_transformer(classifier, test_loader, device=device, prediction_index=-1)
        print(f"[pretrain] test accuracy (all {num_patches} patches): {100 * acc:.2f}  macro-F1: {f1:.3f}")
        print(f"[pretrain] saved {cls_path}")

    # ------------------------------------------------------------------ stage 2: rl
    if "rl" in args.stages:
        print(f"\n=== [rl] PPO patch selector, seed {args.seed} ===")
        seed_everything(seed=args.seed)
        _, val_loader, test_loader, _ = data_loaders()
        classifier = frozen_classifier()

        # trained on D3 (val_loader); test_loader is used to select the best checkpoint
        env = WaferMapGrid(classifier, val_loader, test_loader, start_policy=0,
                           reward_fun=reward_fun, device=device)
        agent = ppo_agent(env, classifier)
        training_loop_ppo_transformer(agent, env, n_epochs=args.rl_epochs)
        print(f"[rl] saved {agent.actor.checkpoint_file}")

    # ------------------------------------------------------------------ stage 3: conformal prediction
    if "cp" in args.stages:
        print(f"\n=== [cp] conformal calibration and evaluation, alpha = {args.alpha} ===")
        seed_everything(seed=args.seed)
        _, val_loader, test_loader, calib_dataset = data_loaders()
        classifier = frozen_classifier()

        env = WaferMapGrid(classifier, val_loader, test_loader, start_policy=0,
                           reward_fun=reward_fun, device=device)
        agent = ppo_agent(env, classifier)
        agent.load_models()

        calib_loader_A, calib_loader_B = make_AB_loaders(
            calib_dataset, batch_size=args.batch_size, split=0.5, shuffle=True, num_workers=4, pin_memory=True)

        cp_kwargs = dict(m=args.m, min_steps=args.min_steps, patience=args.patience,
                         max_steps=None, break_ties_during_run=False)

        cnn_model = None
        if args.hybrid:
            cnn_path = args.cnn_checkpoint or os.path.join(args.models_dir, dataset_name,
                                                           f"resnet34_seed{args.seed}.pth")
            cnn_model = init_resnet("resnet34", num_classes=num_classes, input_channels=input_channels,
                                    pretrained=False)
            cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
            cnn_model = cnn_model.to(device)
            cnn_model.eval()

        def calibrate(alpha):
            _, _, q0_B, q1_B = calibrate_multilabel_CP_two_rule_AB(
                agent=agent, env=env, calib_loader_A=calib_loader_A, calib_loader_B=calib_loader_B,
                alpha=alpha, device=device, **cp_kwargs)
            return q0_B, q1_B

        def evaluate(q0, q1, cnn):
            return evaluate_multilabel_CP_two_rule(
                agent=agent, env=env, test_loader=test_loader, q0=q0, q1=q1, cnn_model=cnn,
                device=device, verbose=False, **cp_kwargs)

        def summary(res):
            return {
                "accuracy": float(res["subset_acc"]),
                "micro_f1": float(res["micro_f1"]),
                "macro_f1": float(res["macro_f1"]),
                "coverage": float(res["coverage"]),
                "scanned_patches_pct": float(100 * res["avg_steps"] / num_patches),
                "unresolved": int(res["unresolved_count"]),
            }

        q0, q1 = calibrate(args.alpha)
        runs = {"non_hybrid": evaluate(q0, q1, None)}
        if args.hybrid:
            runs["hybrid"] = evaluate(q0, q1, cnn_model)

        defect_names = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc", "Near_Full", "Scratch", "Random"]
        report = {"alpha": args.alpha, "model_size": args.model_size, "seed": args.seed}
        for name, res in runs.items():
            Y_true, Y_pred = res["Y_true"], res["Y_pred"]
            report[name] = summary(res)
            report[name]["per_defect"] = {
                d: {"accuracy": float((Y_true[:, i] == Y_pred[:, i]).mean()),
                    "f1": float(f1_score(Y_true[:, i], Y_pred[:, i], zero_division=0))}
                for i, d in enumerate(defect_names)
            }
            with open(os.path.join(args.results_dir, f"cp_{name}_{tag}_alpha{args.alpha}.pickle"), "wb") as fh:
                pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)

        for name in runs:
            s = report[name]
            print(f"[cp] {name:10s} accuracy {s['accuracy']:.2f}%  macro-F1 {s['macro_f1']:.3f}  "
                  f"scanned {s['scanned_patches_pct']:.1f}%  coverage {100 * s['coverage']:.2f}%")

        if args.alpha_sweep:
            sweep = {"alpha": [], "non_hybrid": []}
            if args.hybrid:
                sweep["hybrid"] = []
            for a in np.linspace(0.05, 0.1, 11):
                print(f"\n[cp] alpha sweep: alpha = {a:.3f}")
                q0_a, q1_a = calibrate(a)
                sweep["alpha"].append(float(a))
                sweep["non_hybrid"].append(summary(evaluate(q0_a, q1_a, None)))
                if args.hybrid:
                    sweep["hybrid"].append(summary(evaluate(q0_a, q1_a, cnn_model)))
            report["alpha_sweep"] = sweep

        out = os.path.join(args.results_dir, f"cp_summary_{tag}_alpha{args.alpha}.json")
        with open(out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\n[cp] summary written to {out}")


if __name__ == "__main__":
    run(parse_args())
