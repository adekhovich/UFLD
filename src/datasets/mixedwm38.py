import os

import numpy as np
import torch
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class MixedWM38(Dataset):
    """MixedWM38 wafer maps (52x52) with multi-hot labels over 8 defect types.

    The ``.npz`` file contains ``arr_0`` (wafer maps with values 0 = background, 1 = normal die,
    2 = defective die) and ``arr_1`` (multi-hot labels). Wafer maps are binarised to
    {0: no defect, 255: defect} before the transforms are applied.

    The data is split into train (``train_size``) and a held-out part, which is split in half into
    validation (``val=True``) and test (``val=False``), all with the same ``random_state``.
    """

    def __init__(self, path='data/MixedWM38.npz', train=True, val=False, transforms=None,
                 input_channels=1, num_classes=8, train_size=0.6, random_state=0):
        super().__init__()
        self.path = os.path.expanduser(path)
        self.train = train
        self.val = val
        self.transforms = transforms
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.X, self.y = self.load_data(random_state=random_state, train_size=train_size)

    def load_data(self, random_state, train_size=0.6):
        data = np.load(self.path)
        wafers = data['arr_0']
        labels = data['arr_1']
        X_train, X_test, y_train, y_test = train_test_split(wafers, labels, train_size=train_size,
                                                            shuffle=True, random_state=random_state)
        if self.train:
            return X_train, y_train

        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5,
                                                        shuffle=True, random_state=random_state)
        if self.val:
            return X_val, y_val
        return X_test, y_test

    def set_transform(self, transforms):
        self.transforms = transforms

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        wafer = self.X[index].astype('float32')
        label = self.y[index]

        wafer = np.where(wafer <= 1, 0, 255)   # keep defective dies only
        wafer = torch.from_numpy(wafer).unsqueeze(0).to(dtype=torch.uint8)

        if self.input_channels == 3:
            wafer = wafer.expand(3, -1, -1)

        if self.transforms is not None:
            wafer = torchvision.transforms.ToPILImage()(wafer)
            wafer = self.transforms(wafer)

        label = torch.from_numpy(label).to(dtype=torch.float)
        return wafer, label
