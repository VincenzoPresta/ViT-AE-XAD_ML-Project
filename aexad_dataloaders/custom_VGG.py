import numpy as np
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import albumentations as A


class CustomVGGAD(Dataset):
    def __init__(self, path, train=True, img_size=224):
        super(CustomVGGAD).__init__()
        self.img_size = img_size
        self.train = train
        #self.transform = self.transform_train() if self.train else self.transform_test()

        if self.train:
            split = 'train'
        else:
            split = 'test'

        x = np.load(os.path.join(path, f'X_{split}.npy')).swapaxes(1, 2).swapaxes(2, 3).astype(np.uint8)#[:,:,:,0]
        y = np.load(os.path.join(path, f'Y_{split}.npy'))
        gt = np.load((os.path.join(path, f'GT_{split}.npy'))) #/ 255.0).astype(np.float32)

        #normal_data = x[y == 0]
        #outlier_data = x[y == 1]

        #self.gt = gt
        self.gt = gt[:, 0].astype(np.uint8)
        self.labels = y
        self.images = x #np.append(normal_data, outlier_data, axis=0)
        #self.out_idx_start = normal_data.shape[0]
        #self.normal_idx = np.argwhere(self.labels == 0).flatten()
        #self.outlier_idx = np.argwhere(self.labels == 1).flatten()

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        #self.mean = (x.reshape((-1, 3)) / 255.).mean(axis=0)
        #self.std = (x.reshape((-1, 3)) / 255.).std(axis=0)

        self.dim = (self.images.shape[3], self.images.shape[1], self.images.shape[2])

        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            #transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])

        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            #transforms.RandomChoice([
            #    transforms.ColorJitter(0.04, 0.04, 0.04, 0.04),
            #    transforms.ColorJitter(0.005, 0.0005, 0.0005, 0.0005),
            #]),
            # transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])

        # Aggiunta 30/06/25
        self.augmentors = [  # A.RandomRotate90(),
            # A.Flip(),
            # A.Transpose(),
            A.OpticalDistortion(p=1.0, distort_limit=1.0),
            A.OneOf([
                #A.IAAAdditiveGaussianNoise(), #RIMUOVERE COMMENTO POI
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
                #A.IAAPiecewiseAffine(p=0.3),
                A.PiecewiseAffine(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                #A.IAASharpen(),
                #A.IAAEmboss(),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3)]

        self.transform_mask = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()])

        # ------------------------------------------------

        if self.train:
            self.transform = train_transform
        else:
            self.transform = test_transform

#
    #def transform_test(self):
    #    composed_transforms = transforms.Compose([
    #        transforms.Resize((self.args.img_size, self.args.img_size)),
    #        transforms.ToTensor(),
    #        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #    return composed_transforms
    ## -----------------------------------------

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        #transform = self.transform
        #image = Image.fromarray(self.images[index])
        image = self.images[index]
        ##image = self.transform(image)
        #sample = {'image': transform(image), 'label': self.labels[index]}
        #if index in self.outlier_idx:
        #    image_label = self.gt[index-self.out_idx_start]
        #else:
        #    image_label = np.zeros_like(self.images[index])
        image_label = self.gt[index]
        
        #if self.train:
        #    aug = self.randAugmenter()
        #    augmentated = aug(image=image, mask=image_label)
        #    image, image_label = augmentated['image'], augmentated['mask']

        sample = {'image': self.transform(image), 'label': self.labels[index], 'gt_label': self.transform_mask(image_label)}
        return sample


    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmentors)), 3, replace=False)
        aug = A.Compose([self.augmentors[aug_ind[0]],
                         self.augmentors[aug_ind[1]],
                         self.augmentors[aug_ind[2]]])
        # aug = A.Compose([self.augmentors[0], A.GridDistortion(p=1.0), self.augmentors[3], self.augmentors[1], self.augmentors[7]])
        return aug

    #def getitem(self, index):
    #    # if index in self.outlier_idx and self.train:
    #    #     transform = self.transform_anomaly
    #    # else:
    #    #     transform = self.transform
    #    #image = Image.fromarray(self.images[index])
    #    image = self.images[index]
    #    if index in self.outlier_idx:
    #        image_label = self.gt[index-self.out_idx_start]
    #    else:
    #        image_label = np.zeros_like(self.images[index])
    #    sample = {'image': image, 'label': self.labels[index], 'gt_label': image_label}
    #    return sample


class CustomVGGAD_AE(Dataset):
    def __init__(self, path, train=True):
        super(CustomVGGAD).__init__()
        self.train = train
        #self.transform = self.transform_train() if self.train else self.transform_test()

        if self.train:
            split = 'train'
        else:
            split = 'test'

        x = np.load(os.path.join(path, f'X_{split}.npy')).astype(np.uint8) #/ 255.0)[:,:,:,0]
        y = np.load(os.path.join(path, f'Y_{split}.npy'))

        self.labels = y
        self.images = x[y==0]

        self.dim = self.images.shape[1:]


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        #image = Image.fromarray(self.images[index], mode='RGB')
        image = self.images[index]
        image = self.transform(image)
        sample = {'image': image}
        return sample
