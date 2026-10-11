import copy

import numpy as np
import torch
import torchvision

from src.datasets.mixedwm38 import MixedWM38


def load_dataset(path="data/MixedWM38.npz", train=True, val=False, input_channels=1, img_size=(52, 52),
                 num_classes=8, train_size=0.6, random_state=0):
    """One MixedWM38 split with its transforms.

    The training and validation splits get random flips and rotations; the test split only resizing and
    normalisation. (``load_mixedwm38_splits`` replaces the validation transforms by the test ones.)
    """
    normalize = torchvision.transforms.Normalize((0.5,) * input_channels, (0.5,) * input_channels)
    if train or val:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomRotation(90),
            torchvision.transforms.ToTensor(),
            normalize,
        ])
    else:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor(),
            normalize,
        ])

    return MixedWM38(path=path, train=train, val=val, transforms=transforms, input_channels=input_channels,
                     num_classes=num_classes, train_size=train_size, random_state=random_state)


def get_loaders(train_dataset, test_dataset, val_dataset=None, calib_dataset=None, batch_size=128):
    """Data loaders for the four subsets; the test loader is not shuffled."""
    def loader(dataset, shuffle):
        if dataset is None:
            return None
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    return (loader(train_dataset, True), loader(test_dataset, False),
            loader(val_dataset, True), loader(calib_dataset, True))


def load_mixedwm38_splits(path, img_size=(52, 52), input_channels=1, num_classes=8,
                          train_size=0.6, random_state=2024, calib_fraction=0.1):
    """Split MixedWM38 into the four subsets used in the paper (Table 4).

    Returns
    -------
    train_dataset : D1, trains the Transformer (training augmentations)
    val_dataset   : D3, trains the patch selector (test-time transforms)
    calib_dataset : D2, conformal calibration; every 10th wafer of the training split (test-time transforms)
    test_dataset  : D4, test set
    """
    common = dict(path=path, input_channels=input_channels, img_size=img_size,
                  num_classes=num_classes, train_size=train_size, random_state=random_state)

    train_dataset_full = load_dataset(train=True, **common)
    val_dataset = load_dataset(train=False, val=True, **common)
    test_dataset = load_dataset(train=False, val=False, **common)
    val_dataset.transforms = test_dataset.transforms

    n = len(train_dataset_full)
    num_calib = int(calib_fraction * n)
    calib_idx = np.arange(0, n, max(n // num_calib, 1))
    train_idx = np.setdiff1d(np.arange(n), calib_idx)

    train_dataset = copy.deepcopy(train_dataset_full)
    train_dataset.X = train_dataset_full.X[train_idx].copy()
    train_dataset.y = train_dataset_full.y[train_idx].copy()

    calib_dataset = copy.deepcopy(test_dataset)
    calib_dataset.X = train_dataset_full.X[calib_idx].copy()
    calib_dataset.y = train_dataset_full.y[calib_idx].copy()

    return train_dataset, val_dataset, calib_dataset, test_dataset
