import argparse
import os.path
from typing import List

import torch
from kornia import gaussian_blur2d
from torch import Tensor
from tqdm import tqdm

from fcdd.datasets.noise import kernel_size_to_std
from fcdd.models.bases import ReceptiveNet
from fcdd.training.super_trainer import SuperTrainer
from custom_train_setup import trainer_setup
import numpy as np
import os.path as pt
from time import time

class Trainer():
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.acc_batches = kwargs.pop('acc_batches', 1)

        self.setup = trainer_setup(**self.kwargs)
        self.trainer = SuperTrainer(**self.setup)

    def train(self, epochs):
        self.trainer.train(epochs)

    def eval(self):
        self.trainer.trainer.net.eval()
        labels, loss, anomaly_scores, imgs, outputs, gtmaps, grads = self.trainer.trainer._gather_data(
                    self.trainer.trainer.test_loader,
                )

        with torch.no_grad():
            ascores, red_ascores, gtmaps = self.get_heatmaps_and_scores(anomaly_scores, gtmaps, grads)
        return ascores, red_ascores, gtmaps, labels

    def get_heatmaps_and_scores(self, ascores: Tensor, gtmaps: Tensor = None,
                     grads: Tensor = None, blur=True) -> dict:
        """
        Computes the ROC curves and the AUC for detection performance.
        Also computes those for the explanation performance if ground-truth maps are available.
        :param ascores: anomaly scores
        :param gtmaps: ground-truth maps (can be None)
        :return:
        """
        # Logging
        print('Computing heatmaps...')
        if torch.isnan(ascores).sum() > 0:
            #self.logger.logtxt('Could not compute test scores, since anomaly scores contain nan values!!!', True)
            print('Could not compute test scores, since anomaly scores contain nan values!!!')
            return None
        std = self.trainer.trainer.gauss_std

        gtmap_roc_res, gtmap_prc_res = None, None
        use_grads = grads is not None
        if gtmaps is not None:
            try:
                print('Computing GT test score...')
                ascores = self.trainer.trainer.reduce_pixelwise_ascore(ascores) if not use_grads else grads
                red_ascores = self.trainer.trainer.reduce_ascore(ascores).tolist()
                #gtmaps = self.trainer.trainer.test_loader.dataset.dataset.get_original_gtmaps_normal_class()
                if isinstance(self.trainer.trainer.net, ReceptiveNet):  # Receptive field upsampling for FCDD nets
                    ascores = self.trainer.trainer.net.receptive_upsample(ascores, std=std)
                # Further upsampling for original dataset size
                ascores = torch.nn.functional.interpolate(ascores, (gtmaps.shape[-2:]))
            except AssertionError as e:
                self.logger.warning(f'Skipped computing the gtmap ROC score. {str(e)}')

        if blur:
            if isinstance(self.trainer.trainer.net, ReceptiveNet):
                r = self.trainer.trainer.net.reception['r']
            elif self.objective == 'hsc':
                r = self.trainer.trainer.net.fcdd_cls(self.trainer.trainer.net.in_shape, bias=True).reception['r']
            elif self.trainer.trainer.objective == 'ae':
                enc = self.trainer.trainer.net.encoder
                if isinstance(enc, ReceptiveNet):
                    r = enc.reception['r']
                else:
                    r = enc.fcdd_cls(enc.in_shape, bias=True).reception['r']
            else:
                raise NotImplementedError()
            r = (r - 1) if r % 2 == 0 else r
            std = std or kernel_size_to_std(r)
            ascores = gaussian_blur2d(ascores, (r,) * 2, (std,) * 2)

        return ascores, red_ascores, gtmaps

def launch(ds_folder, epochs, batch_size):

    arguments = {
        'logdir': pt.join('results', 'fcdd_{t}'),#'..', '..', 'data', 'results', 'fcdd_{t}'),
        #'logdir_suffix': '',
        'datadir': ds_folder,
        'objective': 'fcdd',
        'batch_size': batch_size,
        'workers': 0,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4, # 1e-6,
        'optimizer_type': 'sgd',
        'scheduler_type': 'lambda',
        'lr_sched_param': [0.985],
        #'load': None,                   # Path to a file that contains a snapshot of the network model. --> vedere se si può togliere
        'dataset': 'custom_our_mvtec',   # Sostituire con quello custom nostro
        'net': 'FCDD_CNN224_VGG_F', #'FCDD_CNN28_W'
        'preproc': 'lcnaug1', # 'aug1'       lcnaug1 <---- usata in tutti gli altri esperimenti
        'acc_batches': 8, # 1
        'bias': True,
        'cuda': True,#True,
        # artificial anomaly settings
        'supervise_mode': 'custom', #noise
        'noise_mode': 'cifar100',
        'oe_limit': np.infty,
        'online_supervision': True,
        'nominal_label': 0,          # Vedere come gestire questa cosa anche nel nostro codice
        # heatmap generation parameters
        'blur_heatmaps': True,
        'gauss_std': 12, # 10
        'quantile': 0.99, # 0.97
        'resdown': 64
        }

    #args = argparse.Namespace(**arguments)
    args = arguments
    trainer = Trainer(args)
    print(trainer.trainer.trainer.train_loader)
    start = time()
    trainer.train(epochs)
    tot_time = time() - start
    del trainer.trainer.trainer.train_loader#, trainer.trainer.trainer.train_loader.train_labels, (trainer.trainer.trainer.train_loader.train_gt), trainer.trainer.trainer.train_loader._train_set

    print(torch.cuda.is_available())

    # Test data in trainer.trainer.test_loader
    ascores, red_scores, gtmaps, labels = trainer.eval()
    ascores, red_scores, gtmaps, labels = ascores.detach().numpy(), red_scores, gtmaps.detach().numpy(), labels
    del trainer
    return ascores, red_scores, gtmaps, labels, tot_time

