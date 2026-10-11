# UFLD: Uncertainty-aware Feedback Loop Diagnosis

Code for *Uncertainty-aware Feedback Loop Diagnosis: integrating reinforcement learning and conformal prediction for reliable wafer defect classification*.

UFLD classifies a wafer map from a **partial, adaptively chosen set of patches** instead of a full scan. The wafer map (52×52) is split into 169 patches of 4×4, and three components work in a feedback loop:

1. **Causal Transformer classifier** predicts the multi-label defect type from the patches acquired so far. It is pretrained on random patch sequences with binary cross-entropy plus a supervised contrastive loss, using a spatial (patch-position) and a temporal (acquisition-step) embedding.
2. **PPO patch selector** picks the next patch to scan from the Transformer state and the mask of unscanned patches.
3. **Conformal-prediction stopping rule** checks, after every step, whether every label is certified by per-label positive/negative thresholds. The thresholds are calibrated on a held-out set in two stages (A: final step, B: recalibration at the induced stopping times). Scanning stops as soon as all labels are certified. If that never happens, the whole wafer is scanned and, in the *hybrid* setting, a ResNet-34 makes the final prediction.

## Repository structure

```
.
├── main.py                         # full single-model pipeline from the command line
├── notebooks/
│   ├── 0-pretrain.ipynb            # stage 1: pretrain the causal Transformer
│   ├── 1-rl.ipynb                  # stage 2: train the PPO patch selector
│   ├── 2-CP_inference_cls.ipynb    # stage 3: conformal calibration, evaluation, alpha sweep
│   └── 1-rl-ensemble.ipynb         # ensemble of single models: calibration and evaluation
├── requirements.txt
└── src/
    ├── config.py                   # model sizes (Table 1) and checkpoint naming
    ├── datasets/                   # MixedWM38 dataset class
    ├── models/
    │   ├── classifier/             # causal Transformer used for pretraining, ResNet baseline
    │   └── patch_selection/        # KV-cached Transformer, Gym environments, PPO agents
    ├── trainer/                    # Transformer and PPO training loops
    └── utils/
        ├── conformal.py            # two-rule conformal calibration (A/B) and early-stopping evaluation
        ├── contrastive_loss.py     # supervised contrastive loss
        ├── dataloader.py           # transforms, data loaders and the D1–D4 split
        ├── device.py               # default device selection
        ├── eval.py                 # policy evaluation during PPO training
        ├── masked_loss.py          # masked categorical distribution for PPO
        └── utils.py                # seeding, patchify, accuracy helpers
```

## Setup

```bash
pip install -r requirements.txt     # Python 3.9+
```

If you need a specific CUDA build of PyTorch, install `torch` and `torchvision` first from the [official index](https://pytorch.org/get-started/locally/). The experiments in the paper ran on a single NVIDIA RTX A4000 (16 GB).

### Data

Download the **MixedWM38** dataset (38,015 wafer maps, 8 defect types, up to 4 defects per wafer) from its [official repository](https://github.com/Junliangwangdhu/WaferMap) (Kaggle or Google Drive links there) and place it at:

```
data/MixedWM38.npz        # arr_0: wafer maps (52x52; 0 = blank, 1 = normal die, 2 = defective die)
                          # arr_1: multi-hot labels (8 basic defect types)
```

Wafer maps are binarised to defective / not defective dies before training. The dataset was introduced in J. Wang et al., "Deformable Convolutional Networks for Efficient Mixed-type Wafer Defect Pattern Recognition," *IEEE Transactions on Semiconductor Manufacturing*, 2020.

The split is done by `load_mixedwm38_splits` in `src/utils/dataloader.py` with a fixed `random_state=2024`, so it is the same for every seed:

| Subset | Size | Used for |
|---|---|---|
| D1 | 20,528 | train the Transformer |
| D2 | 2,281 | conformal calibration (split 50/50 into A and B) |
| D3 | 7,603 | train the patch selector |
| D4 | 7,603 | test |

Training augmentations (random horizontal/vertical flips and rotations in [−90°, 90°]) are only applied to D1.

## Quick start

`main.py` runs the full single-model pipeline: pretraining, PPO training, then conformal calibration and test evaluation.

```bash
# full pipeline, large Transformer, seed 4
python main.py --seed 4 --model-size large

# calibration and evaluation only, from existing checkpoints,
# with the ResNet-34 fallback and the alpha sweep (Table 5)
python main.py --stages cp --hybrid --alpha-sweep

# quick end-to-end check (1 epoch per training stage)
python main.py --cls-epochs 1 --rl-epochs 1
```

Stages not listed in `--stages` are not re-run; their checkpoints are loaded from `--models-dir`. Each stage re-seeds at its start, so running the stages in one call gives the same result as running them one by one.

### Main options

Run `python main.py --help` for the full list. The defaults are the settings used in the paper.

| Option | Default | Description |
|---|---|---|
| `--stages` | `pretrain rl cp` | stages to run |
| `--seed` | `4` | random seed (also part of the checkpoint names) |
| `--model-size` | `large` | `small` (128-dim, 8 blocks), `medium` (256, 4) or `large` (256, 8) |
| `--data-path` | `data/MixedWM38.npz` | dataset location |
| `--device` | `cuda:0` if available, else `cpu` | device for all models and data |
| `--models-dir` / `--results-dir` | `models` / `results` | where checkpoints and results are written |
| `--cls-epochs` / `--cls-lr` | `240` / `1e-3` | Transformer pretraining (Adam, cosine schedule) |
| `--supcon-weight` | `0.1` | weight of the supervised contrastive loss |
| `--rl-epochs` / `--rl-lr` | `100` / `1e-3` | PPO training |
| `--alpha` | `0.065` | miscoverage level of the conformal stopping rule |
| `--hybrid` | off | also evaluate with the ResNet-34 fallback |
| `--cnn-checkpoint` | `models/mixedwm38/resnet34_seed<seed>.pth` | ResNet-34 weights for `--hybrid` |
| `--alpha-sweep` | off | also evaluate alpha from 0.05 to 0.10 in steps of 0.005 |

### Outputs

| File | Written by |
|---|---|
| `models/mixedwm38_c0.1_supcontransformer_..._seed<seed>.pth` | pretrain (best validation accuracy) |
| `models/actor-ppo_bs16_..._lr0.001_gamma0.99_gae0.99_seed<seed>.pt` (and `critic-...`) | rl |
| `results/cp_non_hybrid_<size>_seed<seed>_alpha<alpha>.pickle` (and `cp_hybrid_...`) | cp: full test predictions, labels and stopping steps |
| `results/cp_summary_<size>_seed<seed>_alpha<alpha>.json` | cp: accuracy, F1, coverage, % scanned patches, per-defect scores, alpha sweep |

Checkpoint names are built by `src/config.py`, so `main.py` and the notebooks read and write the same files.

## Notebooks

The notebooks in `notebooks/` run the same stages interactively. The first cell switches to the repository root, so `data/`, `models/` and `results/` resolve as above. Each notebook has one configuration cell with `SEED`, `MODEL_SIZE` and, for the training notebooks, `TRAIN` (with `TRAIN = False` the existing checkpoint is loaded). Use the same `SEED` and `MODEL_SIZE` across stages.

1. **`0-pretrain.ipynb`** trains and evaluates the causal Transformer.
2. **`1-rl.ipynb`** trains the PPO patch selector on the frozen Transformer.
3. **`2-CP_inference_cls.ipynb`** runs the A/B calibration, evaluates on the test set (including per-defect accuracy and F1) and runs the alpha sweep. Run the sweep once with `HYBRID = False` and then with `HYBRID = True`.

### Ensemble

The ensemble combines independently trained single models: run stages 1 and 2 for each seed in `SEEDS` (default 0–4, `MODEL_SIZE = 'small'`), then run **`1-rl-ensemble.ipynb`**. At every step it averages the members' predicted probabilities and patch-selection distributions, and applies the conformal stopping rule to the averaged probabilities.

### Hybrid setting

With `--hybrid` (or `HYBRID = True` in the notebooks), wafers that are never certified are classified by a ResNet-34 on the full wafer map. The ResNet-34 baseline checkpoint is not included in this repository; train it separately and pass it with `--cnn-checkpoint` (or place it at `models/mixedwm38/resnet34_seed<seed>.pth`).

## Results

MixedWM38 test set; accuracy is the exact match of all eight labels. Single-model results are averaged over five seeds.

| Model | Accuracy (%) | Scanned patches (%) |
|---|---|---|
| ResNet-34 (full scan) | 98.79 ± 0.09 | 100 |
| UFLD-large, non-hybrid | 97.88 ± 0.07 | 61.4 ± 1.6 |
| UFLD-large, hybrid | 98.38 ± 0.10 | 61.4 ± 1.6 |
| Ensemble UFLD-small, non-hybrid | 97.82 | 59.7 |
| Ensemble UFLD-small, hybrid | 98.37 | 59.7 |

To reproduce the single-model rows, run `python main.py --model-size large --hybrid` for seeds 0–4 and average the summaries. See the paper for the comparison with SupConWI-RL, the ablation study and the sensitivity to `alpha`.

## License

This code is released under the MIT License (see `LICENSE.txt`). It includes code adapted from other projects; their licenses and attributions are listed in `THIRD_PARTY_NOTICES.md`.

## Citation

Citation information will be added after the review period.
