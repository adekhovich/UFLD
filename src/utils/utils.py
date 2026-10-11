import os
import random

import numpy as np
import torch
from sklearn.metrics import f1_score


def seed_everything(seed=0):
    """Fix the Python, NumPy and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


def img_to_patch(x, patch_size=4, flatten_channels=True):
    """Split images into a sequence of non-overlapping patches (row-major order).

    Adapted from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html

    Parameters
    ----------
    x : torch.Tensor [B, C, H, W]
    patch_size : int
        Number of pixels per side of a patch.
    flatten_channels : bool
        Return each patch as a flat vector [B, H'*W', C*p*p] instead of [B, H'*W', C, p, p].
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)   # [B, H', W', C, p, p]
    x = x.flatten(1, 2)               # [B, H'*W', C, p, p]
    if flatten_channels:
        x = x.flatten(2, 4)           # [B, H'*W', C*p*p]
    return x


@torch.no_grad()
def accuracy_transformer(model, loader, device, prediction_index=-1):
    """Exact-match accuracy and macro F1 of the causal Transformer after ``prediction_index + 1`` patches.

    ``prediction_index=-1`` uses the full sequence. With ``permute_patches=True`` the patch order is random,
    so seed before calling for reproducible numbers.
    """
    model.eval()
    correct, total = 0, 0
    y_preds, labels = [], []

    for X, y_true in loader:
        X = X.to(device)
        y_true = y_true.to(device)

        y_hat, _, _ = model(X, prediction_index=prediction_index)
        preds = (torch.sigmoid(y_hat) > 0.5).float()

        correct += (preds == y_true).all(dim=1).sum().item()
        total += y_true.size(0)
        y_preds.append(preds.cpu().numpy())
        labels.append(y_true.cpu().numpy())

    f1 = f1_score(np.concatenate(labels), np.concatenate(y_preds), average='macro')
    return correct / max(total, 1), f1
