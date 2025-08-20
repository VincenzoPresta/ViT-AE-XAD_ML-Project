import PIL.Image as Image

import numpy as np
import os
import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms

class MvtecAD(Dataset):
    def __init__(self, path, seed=29, train=True):
        super(MvtecAD).__init__()

        self.train = train
        self.seed = seed
        self.dim = (3, 448, 448)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.dim[-2], self.dim[-1]), Image.NEAREST),
            #transforms.ToTensor(),
            transforms.PILToTensor()
        ])

        if self.train:
            split = 'train'
        else:
            split = 'test'

        x = np.load(os.path.join(path, f'X_{split}.npy'))  # / 255.0)[:,:,:,0]
        y = np.load(os.path.join(path, f'Y_{split}.npy'))
        gt = np.load(os.path.join(path, f'GT_{split}.npy'))

        # normal_data = x[y == 0]
        # outlier_data = x[y == 1]

        self.gt = gt
        self.labels = y
        self.images = x


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image_label = self.gt[index]

        img = self.transform(image)

        print(img.min(), img.max())

        sample = {'image': img / 255.0, 'label': self.labels[index],
                  'gt_label': self.transform(image_label) / 255.0}
        return sample
