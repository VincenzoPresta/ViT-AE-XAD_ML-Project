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

        # === TRANSFORM ORIGINALE (loro) ===
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.dim[-2], self.dim[-1]), Image.NEAREST),   # <-- come l'originale
            transforms.PILToTensor()                                         # <-- come l'originale
        ])

        # === AUGMENTATION (solo per img, non per GT) ===
        if self.train:
            self.aug = get_vit_augmentation(224)
        else:
            self.aug = get_vit_test_transform(224)

        # === LOAD .NPY DATA ===
        split = 'train' if train else 'test'

        self.images = np.load(os.path.join(path, f"X_{split}.npy"))
        self.labels = np.load(os.path.join(path, f"Y_{split}.npy"))
        self.gt     = np.load(os.path.join(path, f"GT_{split}.npy"))

        print(f"Loaded: {self.images.shape}, GT: {self.gt.shape}, labels: {self.labels.shape}")


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):

        img_np = self.images[index]
        gt_np  = self.gt[index]

        # === LORO: convertono a PIL, resize, PILToTensor ===
        img = self.base_transform(img_np)          # [C,H,W] uint8
        gt  = self.base_transform(gt_np) / 255.0   # normalizzato come loro

        # === AUGMENTATION SOLO SULL'IMMAGINE ===
        # ma get_vit_augmentation accetta PIL, quindi torniamo a PIL!
        img_pil = transforms.ToPILImage()(img)     # <-- PIL
        img = self.aug(img_pil)                    # <-- transform finale, inclusa ToTensor+Norm

        sample = {
            'image': img,                # float32 normalizzato [-1,1] (post augment)
            'label': self.labels[index],
            'gt_label': gt               # [0,1] come loro
        }
        return sample
