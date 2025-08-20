import random
import os
import os.path as pt
import numpy as np
import torchvision.transforms as transforms
import torch
import PIL.Image as Image
from typing import Tuple, List
from torch import Tensor
from torch.utils.data import Subset, DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_tensor, to_pil_image
from fcdd.datasets.bases import TorchvisionDataset, GTSubset, GTMapADDataset
from fcdd.datasets.online_supervisor import OnlineSupervisor
from fcdd.datasets.preprocessing import get_target_label_idx, MultiCompose
from fcdd.util.logging import Logger


def extract_custom_classes(datapath: str) -> List[str]:
    dir = os.path.join(datapath, 'custom', 'test')
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    return classes


class ADImageDatasetGTM(TorchvisionDataset):
    base_folder = 'custom'
    ovr = False

    def __init__(self, root: str, normal_class: int, preproc: str, nominal_label: int,
                 supervise_mode: str, noise_mode: str, oe_limit: int, online_supervision: bool,
                 logger: Logger = None):
        """
        :param root: root directory where data is found.
        :param normal_class: the class considered normal.
        :param preproc: the kind of preprocessing pipeline.
        :param nominal_label: the label that marks normal samples in training. The scores in the heatmaps always
            rate label 1, thus usually the normal label is 0, s.t. the scores are anomaly scores.
        :param supervise_mode: the type of generated artificial anomalies.
            See :meth:`fcdd.datasets.bases.TorchvisionDataset._generate_artificial_anomalies_train_set`.
        :param noise_mode: the type of noise used, see :mod:`fcdd.datasets.noise_mode`.
        :param oe_limit: limits the number of different anomalies in case of Outlier Exposure (defined in noise_mode).
        :param online_supervision: whether to sample anomalies online in each epoch,
            or offline before training (same for all epochs in this case).
        :param logger: logger.
        """
        assert online_supervision, 'Artificial anomaly generation for custom datasets needs to be online'
        super().__init__(root, logger=logger)

        self.n_classes = 2
        self.normal_classes = tuple([0])
        self.outlier_classes = [1]
        assert nominal_label in [0, 1]
        self.nominal_label = nominal_label
        self.anomalous_label = 1 if self.nominal_label == 0 else 0

        # ----------------------------- Aggiunte ora --------------------------------
        self.train_images = np.load(os.path.join(root, 'X_train.npy'))
        self.train_labels = np.load(os.path.join(root, 'Y_train.npy'))
        #img_shape = self.train_images.shape
        #self.train_gt = np.full((img_shape[0], 1, img_shape[2], img_shape[3]), fill_value=-1)
        #self.train_gt[self.train_labels == 1] = np.load(os.path.join(root, 'GT_train.npy'))
        self.train_gt = np.load(os.path.join(root, 'GT_train.npy'))

        self.test_images = np.load(os.path.join(root, 'X_test.npy'))
        self.test_labels = np.load(os.path.join(root, 'Y_test.npy'))
        #img_shape = self.test_images.shape
        #self.test_gt = np.full((img_shape[0], 1, img_shape[2], img_shape[3]), fill_value=-1)
        #self.test_gt[self.test_labels == 1] = np.load(os.path.join(root, 'GT_test.npy'))
        self.test_gt = np.load(os.path.join(root, 'GT_test.npy'))

        if self.train_images.shape[1] == 1:
            img_shape = self.train_images.shape
            self.train_images = np.full((img_shape[0], 3, img_shape[2], img_shape[3]), fill_value=self.train_images)
            img_shape = self.test_images.shape
            self.test_images = np.full((img_shape[0], 3,  img_shape[2], img_shape[3]), fill_value=self.test_images)
        # ---------------------------------------------------------------------------

        self.raw_shape = self.train_images.shape[1:]
        self.shape = (3, 224, 224)
        #self.shape = (3, 448, 448)

        # precomputed mean and std of your training data
        self.mean, self.std = self.extract_mean_std(normal_class)

        # img_gtm transforms transform images and corresponding ground-truth maps jointly.
        # This is critically required for random geometric transformations as otherwise
        # the maps would not match the images anymore.
        if preproc in ['', None, 'default', 'none']:
            img_gtm_test_transform = img_gtm_transform = MultiCompose([
                transforms.Resize((self.shape[-2], self.shape[-1]), Image.NEAREST),
                transforms.ToTensor(),
            ])
            test_transform = transform = transforms.Compose([
                transforms.Normalize(self.mean, self.std)
            ])
        elif preproc in ['aug1']:
            img_gtm_transform = MultiCompose([
                transforms.RandomChoice([
                    MultiCompose([
                        #transforms.Resize((self.raw_shape[-2], self.raw_shape[-1]), Image.NEAREST),
                        transforms.Resize((self.shape[-2], self.shape[-1]), Image.NEAREST),
                        transforms.RandomCrop((self.shape[-2], self.shape[-1]), Image.NEAREST),
                    ]),
                    transforms.Resize((self.shape[-2], self.shape[-1]), Image.NEAREST),
                ]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            img_gtm_test_transform = MultiCompose(
                [transforms.Resize((self.shape[-2], self.shape[-1]), Image.NEAREST), transforms.ToTensor()]
            )
            test_transform = transforms.Compose([
                transforms.Normalize(self.mean, self.std)
            ])
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + 0.001 * torch.randn_like(x)),
                transforms.Normalize(self.mean, self.std)
            ])
        #  here you could define other pipelines with augmentations
        else:
            raise ValueError('Preprocessing pipeline {} is not known.'.format(preproc))

        self.target_transform = transforms.Lambda(
            lambda x: self.anomalous_label if x in self.outlier_classes else self.nominal_label
        )

        if supervise_mode not in ['unsupervised', 'other', 'custom']:
            self.all_transform = OnlineSupervisor(self, supervise_mode, noise_mode, oe_limit)
        else:
            self.all_transform = None

        self._train_set = ImageFolderDatasetGTM(
            self.train_images, self.train_labels, self.train_gt, supervise_mode, self.raw_shape, self.ovr,
            self.nominal_label, self.anomalous_label,
            normal_classes=self.normal_classes,
            transform=transform, target_transform=self.target_transform,
            all_transform=self.all_transform,
            img_gtm_transform=img_gtm_transform
        )
        if supervise_mode == 'other':  # (semi)-supervised setting
            self.balance_dataset(gtm=True)
        #else:
        elif supervise_mode != 'custom':
            self._train_set = GTSubset(
                self._train_set, np.argwhere(
                    (np.asarray(self._train_set.anomaly_labels) == self.nominal_label) *
                    np.isin(self._train_set.targets, self.normal_classes)
                ).flatten().tolist()
            )

        self._test_set = ImageFolderDatasetGTM(
            self.test_images, self.test_labels, self.test_gt, supervise_mode, self.raw_shape, self.ovr,
            self.nominal_label, self.anomalous_label,
            normal_classes=self.normal_classes,
            transform=test_transform, target_transform=self.target_transform,
            img_gtm_transform=img_gtm_test_transform
        )
        #if not self.ovr:
        #    self._test_set = GTSubset(
        #        self._test_set, get_target_label_idx(self._test_set.targets, np.asarray(self.normal_classes))
        #    )

    def balance_dataset(self, gtm=False):
        nominal_mask = (np.asarray(self._train_set.anomaly_labels) == self.nominal_label)
        nominal_mask = nominal_mask * np.isin(self._train_set.targets, np.asarray(self.normal_classes))
        anomaly_mask = (np.asarray(self._train_set.anomaly_labels) != self.nominal_label)
        anomaly_mask = anomaly_mask * (1 if self.ovr else np.isin(
            self._train_set.targets, np.asarray(self.normal_classes)
        ))

        if anomaly_mask.sum() == 0:
            return

        CLZ = Subset if not gtm else GTSubset
        self._train_set = CLZ(  # randomly pick n_nominal anomalies for a balanced training set
            self._train_set, np.concatenate([
                np.argwhere(nominal_mask).flatten().tolist(),
                np.random.choice(np.argwhere(anomaly_mask).flatten().tolist(), nominal_mask.sum(), replace=True)
            ])
        )

    def extract_mean_std(self, cls: int) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        transform = transforms.Compose([
            transforms.Resize((self.shape[-2], self.shape[-1])),
            transforms.ToTensor(),
        ])
        ds = ImageFolderDataset(
            self.train_images, self.train_labels,
            'unsupervised', self.raw_shape, self.ovr,
            self.nominal_label, self.anomalous_label,
            normal_classes=[cls],
            transform=transform,
            target_transform=transforms.Lambda(
                lambda x: self.anomalous_label if x in self.outlier_classes else self.nominal_label
            )
        )
        ds = Subset(
            ds,
            np.argwhere(
                np.isin(ds.targets, np.asarray([cls])) * np.isin(ds.anomaly_labels, np.asarray([self.nominal_label]))
            ).flatten().tolist()
        )
        loader = DataLoader(dataset=ds, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
        all_x = []
        for x, _ in loader:
            all_x.append(x)
        all_x = torch.cat(all_x)
        return all_x.permute(1, 0, 2, 3).flatten(1).mean(1), all_x.permute(1, 0, 2, 3).flatten(1).std(1)


class ImageFolderDatasetGTM(GTMapADDataset):
    def __init__(self, imgs, labels, gts, supervise_mode: str, raw_shape: Tuple[int, int, int], ovr: bool,
                 nominal_label: int, anomalous_label: int,
                 transform=None, target_transform=None,
                 normal_classes=None,
                 all_transform=None,
                 img_gtm_transform=None):
        # TODO vedere se possono stare l'init o se devono essere spostati fuori
        #super().__init__(transform=transform, target_transform=target_transform)
        #super().__init__(
        #    root, supervise_mode, raw_shape, ovr, nominal_label, anomalous_label, transform, target_transform,
        #    normal_classes, all_transform
        #)
        self.transform = transform
        self.target_transform = target_transform
        self.all_transform = all_transform

        self.nominal_label = nominal_label
        self.anomalous_label = anomalous_label
        self.normal_classes = normal_classes

        self.images = imgs
        self.targets = labels
        self.anomaly_labels = labels
        self.gts = gts
        # if ovr:
        #     self.anomaly_labels = [self.target_transform(t) for t in self.labels]
        # else:
        #     self.anomaly_labels = [
        #         nominal_label if f.split(os.sep)[-2].lower() in ['normal', 'nominal'] else anomalous_label
        #         for f, _ in self.samples
        #     ]

        #self.normal_classes = normal_classes
        self.all_transform = all_transform      # contains the OnlineSupervisor
        self.supervise_mode = supervise_mode
        self.raw_shape = torch.Size(raw_shape)
        self.img_gtm_transform = img_gtm_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Tensor, int, Tensor]:
        target = self.anomaly_labels[index]
        # TODO vedere come caricare la gt
        #if target == 1:
        gt = torch.from_numpy(self.gts[index]).mul(255).byte()
        gt = to_pil_image(gt)
        #else:
        #   gt = None

        if self.target_transform is not None:
            pass  # already applied since we use self.anomaly_labels instead of self.targets

        if self.all_transform is not None:
            replace = random.random() < 0.5
            if replace:
                if self.supervise_mode not in ['malformed_normal', 'malformed_normal_gt']:
                    img, _, target = self.all_transform(torch.empty(self.raw_shape), None, target, replace=replace)
                else:
                    img = self.images[index]
                    img = to_tensor(img).mul(255).byte()
                    img, gt, target = self.all_transform(img, None, target, replace=replace)
                img = to_pil_image(img)
                gt = gt.mul(255).byte() if gt is not None and gt.dtype != torch.uint8 else gt
                gt = to_pil_image(gt) if gt is not None else None
            else:
                #path, _ = self.samples[index]
                #gt_path, _ = self.gtm_samples[index]
                img = torch.from_numpy(self.images[index])
                img = to_pil_image(img)
        else:
            #path, _ = self.samples[index]
            #gt_path, _ = self.gtm_samples[index]
            #img = self.loader(path)
            img = torch.from_numpy(self.images[index]).mul(255).byte()
            img = to_pil_image(img)
            #if gt_path is not None:
            #    gt = self.loader(gt_path)

        if gt is None:
            # gt is assumed to be 1 for anoms always (regardless of the anom_label), since the supervisors work that way
            # later code fixes that (and thus would corrupt it if the correct anom_label is used here in swapped case)
            gtinitlbl = target if self.anomalous_label == 1 else (1 - target)
            gt = (torch.ones(self.raw_shape)[0] * gtinitlbl).mul(255).byte()
            gt = to_pil_image(gt)

        if self.img_gtm_transform is not None:
            img, gt = self.img_gtm_transform((img, gt))

        if self.transform is not None:
            img = self.transform(img)

        #if self.nominal_label != 0:
        #    gt[gt == 0] = -3  # -3 is chosen arbitrarily here
        #    gt[gt == 1] = self.anomalous_label
        #    gt[gt == -3] = self.nominal_label

        #gt = gt[:1]  # cut off redundant channels
        print('--------', gt.max())

        return img, target, gt


class ImageFolderDataset(Dataset):
    def __init__(self, imgs, labels, supervise_mode: str, raw_shape: Tuple[int, int, int], ovr: bool,
                 nominal_label: int, anomalous_label: int,
                 transform=None,  target_transform=None,
                 normal_classes=None,
                 all_transform=None):
        # TODO vedere se possono stare l'init o se devono essere spostati fuori
        # super().__init__(transform=transform, target_transform=target_transform)
        self.transform = transform
        self.target_transform = target_transform

        self.images = imgs
        self.anomaly_labels = labels
        self.targets = labels

        self.nomina_label = nominal_label
        self.anomalous_label = anomalous_label
        self.normal_classes = normal_classes

        # if ovr:
        #     self.anomaly_labels = [self.target_transform(t) for t in self.labels]
        # else:
        #     self.anomaly_labels = [
        #         nominal_label if f.split(os.sep)[-2].lower() in ['normal', 'nominal'] else anomalous_label
        #         for f, _ in self.samples
        #     ]

        # self.normal_classes = normal_classes
        self.all_transform = all_transform  # contains the OnlineSupervisor
        self.supervise_mode = supervise_mode
        self.raw_shape = torch.Size(raw_shape)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        target = self.anomaly_labels[index]

        if self.target_transform is not None:
            pass  # already applied since we use self.anomaly_labels instead of self.targets

        if self.all_transform is not None:
            replace = random.random() < 0.5
            if replace:
                if self.supervise_mode not in ['malformed_normal', 'malformed_normal_gt']:
                    img, _, target = self.all_transform(
                        torch.empty(self.raw_shape), None, target, replace=replace
                    )
                else:
                    img = torch.from_numpy(self.images[index])
                    img = img.mul(255).byte()
                    img, _, target = self.all_transform(img, None, target, replace=replace)
                img = to_pil_image(img)
            else:
                img = torch.from_numpy(self.images[index])
                img = img.mul(255).byte()
                img = to_pil_image(img)

        else:
            img = torch.from_numpy(self.images[index])
            img = img.mul(255).byte()
            img = to_pil_image(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
