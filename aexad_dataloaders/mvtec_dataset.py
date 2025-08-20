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

        print('Loading image data...')
        x_os = np.load(os.path.join(path, f'X_{split}.npy'))  # / 255.0)[:,:,:,0]
        print('Reshaping image data...')
        x = np.empty((x_os.shape[0], self.dim[0], self.dim[1], self.dim[2]), dtype=x_os.dtype)
        for i in range(x_os.shape[0]):
            x[i] = self.transform(x_os[i])
        del x_os
        print('Done')

        y = np.load(os.path.join(path, f'Y_{split}.npy'))

        print('Loading gt data...')
        gt_os = np.load(os.path.join(path, f'GT_{split}.npy'))
        print('Reshaping gt data...')
        gt = np.empty((gt_os.shape[0], self.dim[0], self.dim[1], self.dim[2]), dtype=gt_os.dtype)
        for i in range(gt_os.shape[0]):
            gt[i] = self.transform(gt_os[i])
        del gt_os
        print('Done')

        # normal_data = x[y == 0]
        # outlier_data = x[y == 1]

        self.gt = gt
        self.labels = y
        self.images = x


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        gt = self.gt[index]

        #img = self.transform(image)
        #gt = self.transform(image_label)

        print('---', img.max(), gt.max())

        sample = {'image': img / 255.0, 'label': self.labels[index],
                  'gt_label': gt / 255.0}
        return sample
