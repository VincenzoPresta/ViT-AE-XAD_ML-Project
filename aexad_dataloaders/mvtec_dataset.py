import PIL.Image as Image
import numpy as np
import os
import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms
from utils.transforms_vit import get_vit_augmentation, get_vit_test_transform


class MvtecAD(Dataset):
    def __init__(self, path, seed=29, train=True):
        super(MvtecAD).__init__()

        self.train = train
        self.seed = seed
        self.dim = (3, 224, 224)

        # === TRANSFORM (solo struttura, apply dopo) ===
        self.base_resize = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224), interpolation=Image.BILINEAR),
        ])

        if self.train:
            self.aug = get_vit_augmentation(224)
        else:
            self.aug = get_vit_test_transform(224)

        # === LOAD X ===
        split = 'train' if train else 'test'
        print('Loading image data...')
        x_np = np.load(os.path.join(path, f'X_{split}.npy'))
        print('Done')

        print('Loading GT data...')
        gt_np = np.load(os.path.join(path, f'GT_{split}.npy'))
        print('Done')

        self.labels = np.load(os.path.join(path, f'Y_{split}.npy'))

        # === SALVO LE IMMAGINI GREZZE (NON trasformate) ===
        self.images = x_np     # shape (N, H, W, C)
        self.gt = gt_np        # shape (N, H, W, C)


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):

        # Recuperiamo immagini grezze (np array)
        img_np = self.images[index]
        gt_np = self.gt[index]

        # Converti a PIL (resize)
        img = self.base_resize(img_np)
        gt = self.base_resize(gt_np)

        # === APPLICA AUGMENTATION SOLO SULL'IMMAGINE ===
        # gt NON va augmentata
        img = self.aug(img)

        # GT → sempre ToTensor + normalizzazione [0,1]
        gt = transforms.ToTensor()(gt)

        # Normalizziamo GT da [0,255] a [0,1] (se non già normalizzata)
        gt = gt / 255.0

        sample = {
            'image': img,
            'label': self.labels[index],
            'gt_label': gt
        }
        return sample
