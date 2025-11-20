import numpy as np
import os
from torch.utils.data import Dataset

class CustomAD(Dataset):
    def __init__(self, path, train=True):
        super(CustomAD).__init__()
        self.train = train
        #self.transform = self.transform_train() if self.train else self.transform_test()

        if self.train:
            split = 'train'
        else:
            split = 'test'

        x = (np.load((os.path.join(path, f'X_{split}.npy'))) / 255.0).astype(np.float32)#[:,:,:,0]
        y = np.load(os.path.join(path, f'Y_{split}.npy'))
        gt = (np.load((os.path.join(path, f'GT_{split}.npy'))) / 255.0).astype(np.float32)

        #normal_data = x[y == 0]
        #outlier_data = x[y == 1]
        print('DS MAX VALUES: ', x.max(), gt.max())

        self.gt = gt
        self.labels = y
        self.images = x #np.append(normal_data, outlier_data, axis=0)
        #self.out_idx_start = normal_data.shape[0]
        #self.normal_idx = np.argwhere(self.labels == 0).flatten()
        #self.outlier_idx = np.argwhere(self.labels == 1).flatten()

        self.dim = self.images.shape[1:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        #transform = self.transform
        #image = Image.fromarray(self.images[index])
        #sample = {'image': transform(image), 'label': self.labels[index]}
        #if index in self.outlier_idx:
        #    image_label = self.gt[index-self.out_idx_start]
        #else:
        #    image_label = np.zeros_like(self.images[index])
        image_label = self.gt[index]
        sample = {'image': self.images[index], 'label': self.labels[index], 'gt_label': image_label}
        return sample



class CustomAD_AE(Dataset):
    def __init__(self, path, train=True):
        super(CustomAD).__init__()
        self.train = train
        #self.transform = self.transform_train() if self.train else self.transform_test()

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
        return len(self.images)

    def __getitem__(self, index):
        sample = {'image': self.images[index]}
        return sample

