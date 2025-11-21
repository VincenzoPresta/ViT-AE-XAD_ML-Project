import PIL.Image as Image
import numpy as np
import os
import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms
from augmentation.transforms_vit import get_vit_augmentation, get_vit_test_transform


class MvtecAD(Dataset):
    def __init__(self, path, seed=29, train=True):
        super(MvtecAD).__init__()

        self.train = train
        self.seed = seed
        self.dim = (3, 224, 224)

        # Resize - sempre uguale per X e GT
        self.base_resize = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224), interpolation=Image.BILINEAR),
        ])

        # ---- AUGMENTATION ----
        if self.train:
            self.aug = get_vit_augmentation(224)
        else:
            self.aug = get_vit_test_transform(224)

        # ---- LOAD .NPY DATA ----
        split = 'train' if train else 'test'

        print('Loading image data...')
        x_np = np.load(os.path.join(path, f'X_{split}.npy'))   # (N,H,W,C)
        print('Done')

        print('Loading GT data...')
        gt_np = np.load(os.path.join(path, f'GT_{split}.npy'))
        print('Done')

        self.labels = np.load(os.path.join(path, f'Y_{split}.npy'))

        self.images = x_np
        self.gt = gt_np


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):

        img_np = self.images[index]
        gt_np = self.gt[index]

        # Resize
        img = self.base_resize(img_np)
        gt  = self.base_resize(gt_np)

        # Img: augmentation (train) o test-transform
        img = self.aug(img)

        # GT: solo ToTensor e normalizzazione
        gt = transforms.ToTensor()(gt)
        gt = gt / 255.0

        return {
            'image': img,
            'label': self.labels[index],
            'gt_label': gt
        }
