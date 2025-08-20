import cv2
import numpy as np
import torch

import argparse
import os

from scipy.ndimage import gaussian_filter

#from custom_localization import get_heatmaps_and_scores

from competitors.deviation.custom_localization import get_heatmaps_and_scores
from competitors.deviation.dataloaders.dataloader import build_dataloader
from competitors.deviation.localization import farward_hook, backward_hook, convert_to_grayscale
from competitors.deviation.modeling.net import SemiADNet
from tqdm import tqdm
from competitors.deviation.utils import aucPerformance
from competitors.deviation.modeling.layers import build_criterion
from time import time
from PIL import Image

class Trainer(object):

    def __init__(self, args):
        self.args = args

        # Define Dataloader
        kwargs = {'num_workers': args.workers}
        self.train_loader, self.test_loader = build_dataloader(args, **kwargs)

        self.model = SemiADNet(args)

        self.criterion = build_criterion(args.criterion)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        self.cuda = args.cuda

        if args.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            tbar = tqdm(self.train_loader)
            train_loss = 0.0
            for i, sample in enumerate(tbar):
                image, target = sample['image'], sample['label']
                #print('TRAIN IMAGE: ', image.shape)
                if self.args.cuda:
                    image, target = image.cuda(), target.cuda()

                output = self.model(image)
                loss = self.criterion(output, target.unsqueeze(1).float())
                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()
                tbar.set_description('Epoch:%d, Train loss: %.3f' % (epoch, train_loss / (i + 1)))
            self.scheduler.step()

    def eval(self):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        total_pred = np.array([])
        total_target = np.array([])
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image.float())
            loss = self.criterion(output, target.unsqueeze(1).float())
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            total_pred = np.append(total_pred, output.data.cpu().numpy())
            total_target = np.append(total_target, target.cpu().numpy())
        roc, pr = aucPerformance(total_pred, total_target)
        return roc, pr

    def save_weights(self, filename):
        torch.save(self.model.state_dict(), os.path.join(self.args.experiment_dir, filename))


# Qua gli dobbiamo passare i dati di addestramento e tutti i parametri del metodo
def launch(dataset_root, epochs=50, batch_size=48, seed=42, img_size=448):
    # Argomenti che nello script originali erano passati dalla riga di comando
    arguments = {
        'batch_size': batch_size,
        'epochs': epochs,
        'ramdn_seed': seed,
        'workers': 4,
        'no_cuda': False,
        'weight_name': 'model.pkl',
        'dataset_root': dataset_root,
        'experiment_dir': './experiment',
        'img_size': img_size,
        'n_anomaly': 0,                     # Vedere se si può eliminare o si deve lasciare per compatibilità
        'n_scales': 2,
        'backbone': 'resnet18',
        'criterion': 'deviation',
        'topk': 0.1,
    }

    args = argparse.Namespace(**arguments)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    trainer = Trainer(args)
    torch.manual_seed(args.ramdn_seed)

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    start = time()
    trainer.train(trainer.args.epochs)
    tot_time = time() - start

    #trainer.eval()
    htmaps, scores, gtmaps, labels = get_heatmaps_and_scores(trainer.model, trainer.test_loader, trainer.cuda)
    return htmaps, scores, gtmaps, labels, tot_time


if __name__ == '__main__':
    ret = launch(dataset_root='competitors/deviation/data/test', epochs=1)