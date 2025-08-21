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
from fcdd.datasets.preprocessing import get_target_label_idx, MultiCompose, local_contrast_normalization
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
                 logger: Logger = None, shape: tuple = (3, 28, 28)):
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

        # min max after gcn l1 norm has been applied
        min_max_l1 = [
            [(-1.3336724042892456, -1.3107913732528687, -1.2445921897888184),
             (1.3779616355895996, 1.3779616355895996, 1.3779616355895996)],
            [(-2.2404820919036865, -2.3387579917907715, -2.2896201610565186),
             (4.573435306549072, 4.573435306549072, 4.573435306549072)],
            [(-3.184587001800537, -3.164201259613037, -3.1392977237701416),
             (1.6995097398757935, 1.6011602878570557, 1.5209171772003174)],
            [(-3.0334954261779785, -2.958242416381836, -2.7701096534729004),
             (6.503103256225586, 5.875098705291748, 5.814228057861328)],
            [(-3.100773334503174, -3.100773334503174, -3.100773334503174),
             (4.27892541885376, 4.27892541885376, 4.27892541885376)],
            [(-3.6565306186676025, -3.507692813873291, -2.7635035514831543),
             (18.966819763183594, 21.64590072631836, 26.408710479736328)],
            [(-1.5192601680755615, -2.2068002223968506, -2.3948357105255127),
             (11.564697265625, 10.976534843444824, 10.378695487976074)],
            [(-1.3207964897155762, -1.2889339923858643, -1.148416519165039),
             (6.854909896850586, 6.854909896850586, 6.854909896850586)],
            [(-0.9883341193199158, -0.9822461605072021, -0.9288841485977173),
             (2.290637969970703, 2.4007883071899414, 2.3044068813323975)],
            [(-7.236185073852539, -7.236185073852539, -7.236185073852539),
             (3.3777384757995605, 3.3777384757995605, 3.3777384757995605)],
            [(-3.2036616802215576, -3.221003532409668, -3.305514335632324),
             (7.022546768188477, 6.115569114685059, 6.310940742492676)],
            [(-0.8915618658065796, -0.8669204115867615, -0.8002046346664429),
             (4.4255571365356445, 4.642300128936768, 4.305730819702148)],
            [(-1.9086798429489136, -2.0004451274871826, -1.929288387298584),
             (5.463134765625, 5.463134765625, 5.463134765625)],
            [(-2.9547364711761475, -3.17536997795105, -3.143850803375244),
             (5.305514812469482, 4.535006523132324, 3.3618252277374268)],
            [(-1.2906527519226074, -1.2906527519226074, -1.2906527519226074),
             (2.515115737915039, 2.515115737915039, 2.515115737915039)]
        ]

        # mean and std of original images per class
        mean = [
            (0.53453129529953, 0.5307118892669678, 0.5491130352020264),
            (0.326835036277771, 0.41494372487068176, 0.46718254685401917),
            (0.6953922510147095, 0.6663950085639954, 0.6533040404319763),
            (0.36377236247062683, 0.35087138414382935, 0.35671544075012207),
            (0.4484519958496094, 0.4484519958496094, 0.4484519958496094),
            (0.2390524297952652, 0.17620408535003662, 0.17206747829914093),
            (0.3919542133808136, 0.2631213963031769, 0.22006843984127045),
            (0.21368788182735443, 0.23478130996227264, 0.24079132080078125),
            (0.30240726470947266, 0.3029524087905884, 0.32861486077308655),
            (0.7099748849868774, 0.7099748849868774, 0.7099748849868774),
            (0.4567880630493164, 0.4711957275867462, 0.4482630491256714),
            (0.19987481832504272, 0.18578395247459412, 0.19361256062984467),
            (0.38699793815612793, 0.276934415102005, 0.24219433963298798),
            (0.6718143820762634, 0.47696375846862793, 0.35050269961357117),
            (0.4014520049095154, 0.4014520049095154, 0.4014520049095154)
        ]
        std = [
            (0.3667600452899933, 0.3666728734970093, 0.34991779923439026),
            (0.15321789681911469, 0.21510766446590424, 0.23905669152736664),
            (0.23858436942100525, 0.2591284513473511, 0.2601949870586395),
            (0.14506031572818756, 0.13994529843330383, 0.1276693195104599),
            (0.1636597216129303, 0.1636597216129303, 0.1636597216129303),
            (0.1688646823167801, 0.07597383111715317, 0.04383210837841034),
            (0.06069392338395119, 0.04061736911535263, 0.0303945429623127),
            (0.1602524220943451, 0.18222476541996002, 0.15336430072784424),
            (0.30409011244773865, 0.30411985516548157, 0.28656429052352905),
            (0.1337062269449234, 0.1337062269449234, 0.1337062269449234),
            (0.12076705694198608, 0.13341768085956573, 0.12879984080791473),
            (0.22920562326908112, 0.21501320600509644, 0.19536510109901428),
            (0.20621345937252045, 0.14321941137313843, 0.11695228517055511),
            (0.08259467780590057, 0.06751163303852081, 0.04756828024983406),
            (0.32304847240448, 0.32304847240448, 0.32304847240448)
        ]

        # ----------------------------- Aggiunte ora --------------------------------
        self.train_images = np.load(os.path.join(root, 'X_train.npy'))
        
        if self.train_images.shape[1] == 1:
            self.train_images = np.repeat(self.train_images, 3, axis=1) # Se il dataset è in scala di grigi (1 canale), replico il canale 3 volte per avere RGB
        
        self.train_labels = np.load(os.path.join(root, 'Y_train.npy'))
        #img_shape = self.train_images.shape
        #self.train_gt = np.full((img_shape[0], 1, img_shape[2], img_shape[3]), fill_value=-1)
        #self.train_gt[self.train_labels == 1] = np.load(os.path.join(root, 'GT_train.npy'))
        anom_ids = list(np.argwhere(self.train_labels == self.anomalous_label).reshape(-1))
        self.train_ids_anom = dict(zip(anom_ids, range(len(anom_ids))))
        self.train_gt = np.load(os.path.join(root, 'GT_train.npy'))[anom_ids]

        self.test_images = np.load(os.path.join(root, 'X_test.npy'))
        
        if self.test_images.shape[1] == 1:
            self.test_images = np.repeat(self.test_images, 3, axis=1)
        
        self.test_labels = np.load(os.path.join(root, 'Y_test.npy'))
        #img_shape = self.test_images.shape
        #self.test_gt = np.full((img_shape[0], 1, img_shape[2], img_shape[3]), fill_value=-1)
        #self.test_gt[self.test_labels == 1] = np.load(os.path.join(root, 'GT_test.npy'))
        anom_ids = list(np.argwhere(self.test_labels == self.anomalous_label).reshape(-1))
        self.test_ids_anom = dict(zip(anom_ids, range(len(anom_ids))))
        self.test_gt = np.load(os.path.join(root, 'GT_test.npy'))[anom_ids]

        # ---------------------------------------------------------------------------

        self.raw_shape = self.train_images.shape[1:]
        self.shape = shape

        if self.raw_shape[0] != self.shape[0]:
        # Se è grayscale, forza a 3 canali
            if self.raw_shape[0] == 1 and self.shape[0] == 3:
                self.train_images = np.repeat(self.train_images, 3, axis=1)
                self.test_images  = np.repeat(self.test_images, 3, axis=1)

        # precomputed mean and std of your training data
        if len(self.train_labels[self.train_labels==normal_class]) > 0:
            self.mean, self.std = self.extract_mean_std(normal_class)
        else:
            self.mean = mean
            self.std = std

        # img_gtm transforms transform images and corresponding ground-truth maps jointly.
        # This is critically required for random geometric transformations as otherwise
        # the maps would not match the images anymore.
        img_gtm_transform, img_gtm_test_transform = None, None
        all_transform = []
        if preproc == 'lcn':
            assert self.raw_shape == self.shape, 'in case of no augmentation, raw shape needs to fit net input shape'
            img_gtm_transform = img_gtm_test_transform = MultiCompose([
                transforms.ToTensor(),
            ])
            test_transform = transform = transforms.Compose([
                transforms.Lambda(lambda x: local_contrast_normalization(x, scale='l1')),
                transforms.Normalize(
                    min_max_l1[normal_class][0],
                    [ma - mi for ma, mi in zip(min_max_l1[normal_class][1], min_max_l1[normal_class][0])]
                )
            ])
        elif preproc in ['', None, 'default', 'none']:
            assert self.raw_shape == self.shape, 'in case of no augmentation, raw shape needs to fit net input shape'
            img_gtm_transform = img_gtm_test_transform = MultiCompose([
                transforms.ToTensor(),
            ])
            test_transform = transform = transforms.Compose([
                transforms.Normalize(mean[normal_class], std[normal_class])
            ])
        elif preproc in ['aug1']:
            img_gtm_transform = MultiCompose([
                transforms.RandomChoice(
                    [transforms.RandomCrop(self.shape[-1], padding=0), transforms.Resize((self.shape[-2], self.shape[-1]), Image.NEAREST)]
                ),
                transforms.ToTensor(),
            ])
            img_gtm_test_transform = MultiCompose(
                [transforms.Resize((self.shape[-2], self.shape[-1]), Image.NEAREST), transforms.ToTensor()]
            )
            test_transform = transforms.Compose([
                transforms.Normalize(mean[normal_class], std[normal_class])
            ])
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomChoice([
                    transforms.ColorJitter(0.04, 0.04, 0.04, 0.04),
                    transforms.ColorJitter(0.005, 0.0005, 0.0005, 0.0005),
                ]),
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: (x + torch.randn_like(x).mul(np.random.randint(0, 2)).mul(x.std()).mul(0.1)).clamp(0, 1)
                ),
                transforms.Normalize(mean[normal_class], std[normal_class])
            ])
        elif preproc in ['lcnaug1']:
            img_gtm_transform = MultiCompose([
                transforms.RandomChoice(
                    [transforms.RandomCrop(self.shape[-1], padding=0), transforms.Resize((self.shape[-2], self.shape[-1]), Image.NEAREST)]
                ),
                transforms.ToTensor(),
            ])
            img_gtm_test_transform = MultiCompose(
                [transforms.Resize((self.shape[-2], self.shape[-1]), Image.NEAREST), transforms.ToTensor()]
            )
            test_transform = transforms.Compose([
                transforms.Lambda(lambda x: local_contrast_normalization(x, scale='l1')),
                transforms.Normalize(
                    min_max_l1[normal_class][0],
                    [ma - mi for ma, mi in zip(min_max_l1[normal_class][1], min_max_l1[normal_class][0])]
                )
            ])
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomChoice([
                    transforms.ColorJitter(0.04, 0.04, 0.04, 0.04),
                    transforms.ColorJitter(0.005, 0.0005, 0.0005, 0.0005),
                ]),
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: (x + torch.randn_like(x).mul(np.random.randint(0, 2)).mul(x.std()).mul(0.1)).clamp(0, 1)
                ),
                transforms.Lambda(lambda x: local_contrast_normalization(x, scale='l1')),
                transforms.Normalize(
                    min_max_l1[normal_class][0],
                    [ma - mi for ma, mi in zip(min_max_l1[normal_class][1], min_max_l1[normal_class][0])]
                )
            ])
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
            self.train_images, self.train_labels, self.train_gt, self.train_ids_anom, supervise_mode, self.raw_shape, self.ovr,
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
            self.test_images, self.test_labels, self.test_gt, self.test_ids_anom, supervise_mode, self.raw_shape, self.ovr,
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
    def __init__(self, imgs, labels, gts, ids_anom, supervise_mode: str, raw_shape: Tuple[int, int, int], ovr: bool,
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
        self.ids_anom = ids_anom
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
        if target == 1:
            gt = self.gts[self.ids_anom[index]]
            if gt.ndim == 2: # MNIST masks are (H,W)
                gt = torch.from_numpy(gt).unsqueeze(0).byte() * 255
            elif gt.ndim == 3 and gt.shape[0] == 1:
                gt = torch.from_numpy(gt).byte() * 255
            else:
                raise ValueError(f"Unexpected GT shape: {gt.shape}")
            gt = to_pil_image(gt)
        else:
            # maschera vuota 1 canale per MNIST
            gt = to_pil_image(np.zeros((28,28), dtype=np.uint8))

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
        #print('----', gt.max())

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
