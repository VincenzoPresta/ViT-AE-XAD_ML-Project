import PIL.Image as Image

import numpy as np
import os
import torch
import random
from torch.utils.data import Dataset

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from dataset import CustomAD


class AugmentedAD(Dataset):
    '''
    Data loader for generic RGB datasets, this data loader supposes to have 3 files
        - X_<split>.npy containing (3 x height x width) images, <split> is supposed to be 'train' or 'test'
        - Y_<split>.npy containing the labels, <split> is supposed to be 'train' or 'test'
        - GT_<split>.npy containing (3 x height x width) ground truth heatmaps, <split> is supposed to be 'train' or 'test', the 3 channels contains the same heatmap repeated 3 times. GT heatmaps for normal images are equal to 0 in every pixel
    '''
    def __init__(self, path, seed=None, train=True):
        '''
        :param path: str, path to the tree files containing the dataset
        :param train: bool, if True the dataloader loads the train dataset, it loads the test set otherwise, default True
        '''
        super(AugmentedAD).__init__()

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

        x = np.load(os.path.join(path, f'X_{split}.npy'))
        y = np.load(os.path.join(path, f'Y_{split}.npy'))
        gt = np.load(os.path.join(path, f'GT_{split}.npy'))


        self.gt = gt
        self.labels = y
        self.images = x


    def __len__(self):
        '''
        Returns the number of samples composing the dataset
        '''
        return len(self.images)

    def __getitem__(self, index):
        '''
        Returns one of the samples of the dataset
        :param index: index of the sample to return
        '''
        image = self.transform(self.images[index])
        image_label = self.transform(self.gt[index])

        # Data augmentation on the flight
        if self.train and random.uniform(0, 1) > 0.2:
            alpha = random.randint(0, 5)
            image = TF.rotate(image, angle=alpha * 90.)
            image_label = TF.rotate(image_label, angle=alpha * 90.)

        sample = {'image': image / 255.0, 'label': self.labels[index],
                  'gt_label': image_label / 255.0}
        return sample

class AugmentedAD_AE(Dataset):
    '''
    Data loader for generic RGB datasets tailored to autoencoder, this data loader supposes to have 2 files
        - X_<split>.npy containing (3 x height x width) images, <split> is supposed to be 'train' or 'test', anomalous images will be ignored
        - Y_<split>.npy containing the labels, <split> is supposed to be 'train' or 'test', labels are used only to filter data
    '''
    def __init__(self, path, train=True):
        super(AugmentedAD).__init__()
        self.train = train
        self.dim = (3, 448, 448)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.dim[-2], self.dim[-1]), Image.NEAREST),
            transforms.PILToTensor()
        ])

        if self.train:
            split = 'train'
        else:
            split = 'test'

        x = np.load(os.path.join(path, f'X_{split}.npy')) #/ 255.0)[:,:,:,0]
        y = np.load(os.path.join(path, f'Y_{split}.npy'))

        self.labels = y
        self.images = x[y==0]

        self.dim = self.images.shape[1:]


    def __len__(self):
        '''
            Returns the number of samples composing the dataset
        '''
        return len(self.images)

    def __getitem__(self, index):
        '''
        Returns one of the samples of the dataset
        :param index: index of the sample to return
        '''

        image = self.images[index]

        sample = {'image': self.transform(image) / 255.0}
        return sample