import numpy as np
import torch
from sklearn.metrics import f1_score

from src.utils.utils import img_to_patch


@torch.no_grad()
def eval_policy_transformer(agent, env, train=False):
    """Roll out the deterministic policy over a full episode (all patches) for every wafer.

    Used during PPO training to select the best checkpoint by discounted reward.

    Parameters
    ----------
    agent : PPOAgent
    env : WaferMapGrid
    train : bool
        Evaluate on ``env.train_loader`` if True, otherwise on ``env.test_loader``.

    Returns
    -------
    acc : float
        Exact-match accuracy (%) after all patches have been acquired.
    f1 : float
        Macro F1 score after all patches have been acquired.
    score : float
        Mean discounted reward per wafer (the checkpoint-selection criterion).
    patch_order : torch.Tensor [N, T]
        Acquisition order chosen by the policy.
    None
        Kept for backward compatibility of the return signature.
    """
    agent.actor.eval()

    data_loader = env.train_loader if train else env.test_loader
    gamma = agent.gamma

    total_reward = 0.0
    correct = 0
    preds, labels, patch_order = [], [], []

    for X, y in data_loader:
        X = X.to(env.device)
        y = y.to(env.device)

        patches = img_to_patch(X, patch_size=env.classifier.patch_size)  # [B, S, input_dim]
        seq_len = patches.size(1)

        _, observation = env.reset(patches=patches, y=y)

        discount = 1.0
        order_batch = []
        logits = None
        for _ in range(seq_len):
            action, _ = agent.choose_action(observation, env.action_mask.clone(), determenistic=True)
            logits, observation, reward, _, _ = env.step(action)

            if env.curr_time >= env.start_policy:
                total_reward += (discount * reward).sum().item()
                discount *= gamma

            order_batch.append(action)

        patch_order.append(torch.stack(order_batch, dim=1).cpu())

        if y.dim() > 1:   # multi-label (MixedWM38): sigmoid + 0.5 threshold
            pred = (torch.sigmoid(logits) >= 0.5).to(y.dtype)
            correct += (pred == y).all(dim=1).sum().item()
        else:             # single-label
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()

        preds.append(pred.cpu().numpy())
        labels.append(y.cpu().numpy())

    n = len(data_loader.dataset)
    acc = 100.0 * correct / n
    f1 = f1_score(np.concatenate(labels), np.concatenate(preds), average="macro", zero_division=0)
    score = total_reward / n

    return acc, f1, score, torch.cat(patch_order, dim=0), None
