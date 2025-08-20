import numpy as np
import os, sys

import torch
from competitors.deviation.datasets.base_dataset import BaseADDataset
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision import transforms

class CustomAD(BaseADDataset):

    def __init__(self, args, train = True):
        super(CustomAD).__init__()
        self.args = args
        self.train = train
        self.transform = self.transform_train() if self.train else self.transform_test()

        self.root = self.args.dataset_root

        if self.train:
            split = 'train'
        else:
            split = 'test'

        self.images = np.load(open(os.path.join(self.root, f'X_{split}.npy'), 'rb'))
        print('IMAGES SHAPE: ', self.images.shape)
        #if self.images.shape[1] != 3:
        if self.images.shape[1] == 1:
            img_shape = self.images.shape
            self.images = np.full((img_shape[0], 3, img_shape[2], img_shape[3]), fill_value=self.images)

        self.original_shape = self.images.shape[2:]

        self.labels = np.load(open(os.path.join(self.root, f'Y_{split}.npy'), 'rb'))
        print('LABELS SHAPE: ', self.labels.shape)
        print('IMG SIZE: ', self.args.img_size)
        self.gt = np.load(open(os.path.join(self.root, f'GT_{split}.npy'), 'rb'))

        self.normal_idx = np.argwhere(self.labels == 0).flatten()
        self.outlier_idx = np.argwhere(self.labels == 1).flatten()

    def transform_train(self):
        composed_transforms = transforms.Compose([
            transforms.Resize((self.args.img_size, self.args.img_size)),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return composed_transforms

    def transform_test(self):
        composed_transforms = transforms.Compose([
            transforms.Resize((self.args.img_size, self.args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return composed_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        transform = self.transform
        image = torch.from_numpy(self.images[index]).byte()
        #image = self.images[index] * 255
        #image = image.astype(np.uint8)
        image = to_pil_image(image)
        sample = {'image': transform(image), 'label': self.labels[index]}
        return sample

    def getitem(self, index):
        if index in self.outlier_idx and self.train:
            transform = self.transform_anomaly
        else:
            transform = self.transform

        #print('IMG: ', self.images[index].max(), self.images[index].min())
        #image = Image.fromarray(self.images[index])
        #image = torch.from_numpy(self.images[index]).mul(255).byte()
        image = torch.from_numpy(self.images[index]).byte()

        #print('IMG2: ', image.max(), image.min())
        image = to_pil_image(image)

        image_label = self.gt[index]

        sample = {'image': transform(image), 'label': self.labels[index], 'seg_label': image_label, 'raw_image':image}
        return sample
