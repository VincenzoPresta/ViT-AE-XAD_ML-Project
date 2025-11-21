import os
import numpy as np
import torch
from torch.utils.data import Dataset


class TensorDatasetAD(Dataset):
    """
    Dataloader generico per dataset salvati in formato NPY:
    - X_train.npy / X_test.npy  (float32, shape N x C x H x W)
    - GT_train.npy / GT_test.npy (float32, shape N x 1 x H x W)
    - Y_train.npy / Y_test.npy  (0/1)

    Nessuna augmentation viene applicata qui.
    """
    def __init__(self, base_path, train=True):
        super().__init__()

        split = "train" if train else "test"

        X_path  = os.path.join(base_path, f"X_{split}.npy")
        Y_path  = os.path.join(base_path, f"Y_{split}.npy")
        GT_path = os.path.join(base_path, f"GT_{split}.npy")

        # Caricamento dataset
        self.images = np.load(X_path).astype(np.float32)     # shape (N,C,H,W)
        self.labels = np.load(Y_path).astype(np.int64)       # 0 / 1
        self.gt      = np.load(GT_path).astype(np.float32)   # shape (N,1,H,W)

        assert self.images.ndim == 4, "X deve essere in formato NCHW"
        assert self.gt.ndim == 4, "GT deve essere in formato NCHW"

        self.train = train

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):

        img = torch.tensor(self.images[idx])   # (C,H,W)
        gt  = torch.tensor(self.gt[idx])       # (1,H,W)
        lab = torch.tensor(self.labels[idx])   # scalar 0/1

        return {
            "image": img,
            "gt_label": gt,
            "label": lab
        }
