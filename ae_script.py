import os

import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from time import time

from AE_architectures import Shallow_Autoencoder, Deep_Autoencoder, Conv_Autoencoder, PCA_Autoencoder
from aexad_dataloaders.dataset import CustomAD, CustomAD_AE


class Trainer_AE:
    def __init__(self, latent_dim, lambda_p, lambda_s, f, path, AE_type, batch_size=None, silent=False, use_cuda=True,
                 save_intermediate=False):
        '''
        :param latent_dim:
        :param lambda_p: anomalous pixel weight
        :param lambda_s: anomalous samples weight
        :param f: mapping function for anomalous pixels
        :param path:
        :param AE_type:
        :param batch_size:
        :param silent:
        :param use_cuda:
        '''
        self.silent = silent
        self.train_loader = DataLoader(CustomAD_AE(path, train=True), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(CustomAD(path, train=False), batch_size=batch_size, shuffle=False)
        self.save_intermediate = save_intermediate

        if lambda_s is None:
            lambda_s = len(self.train_loader.dataset) / np.sum(self.train_loader.dataset.labels)

        self.cuda = use_cuda and torch.cuda.is_available()

        if AE_type == 'shallow':
            self.model = Shallow_Autoencoder(self.train_loader.dataset.dim, np.prod(self.train_loader.dataset.dim),
                                             latent_dim)
        # deep
        elif AE_type == 'deep':
            self.model = Deep_Autoencoder(self.train_loader.dataset.dim, flat_dim=np.prod(self.train_loader.dataset.dim),
                                          intermediate_dim=256, latent_dim=latent_dim)
        # conv
        elif AE_type == 'conv':
            self.model = Conv_Autoencoder(self.train_loader.dataset.dim)

        elif AE_type == 'pca':
            self.model = PCA_Autoencoder(np.prod(self.train_loader.dataset.dim), np.prod(self.train_loader.dataset.dim),
                                         latent_dim)
        else:
            raise Exception()

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()

        if self.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()



    def test(self):
        self.model.eval()
        tbar = tqdm(self.test_loader, disable=self.silent)
        shape = [0]
        shape.extend(self.test_loader.dataset.images.shape[1:])
        heatmaps = []
        scores = []
        gtmaps = []
        labels = []
        for i, sample in enumerate(tbar):
            image, label, gtmap = sample['image'], sample['label'], sample['gt_label']
            # TODO ricontrollare
            if self.cuda:
                image = image.cuda()
            output = self.model(image).detach().cpu().numpy()
            image = image.cpu().numpy()
            heatmap = ((image-output) ** 2)#.numpy()
            score = heatmap.reshape((image.shape[0], -1)).mean(axis=-1)
            heatmaps.extend(heatmap)
            scores.extend(score)
            gtmaps.extend(gtmap.detach().numpy())
            labels.extend(label.detach().numpy())
        scores = np.array(scores)
        heatmaps = np.array(heatmaps)
        gtmaps = np.array(gtmaps)
        labels = np.array(labels)
        return heatmaps, scores, gtmaps, labels


    def train(self, epochs, save_path=''):
        if isinstance(self.model, Conv_Autoencoder):
            name = 'ae_model_conv'
        elif isinstance(self.model, Deep_Autoencoder):
            name = 'ae_model_deep'
        elif isinstance(self.model, Shallow_Autoencoder):
            name = 'ae_model'

        self.model.train()
        for epoch in range(epochs):
            train_loss = 0.0
            tbar = tqdm(self.train_loader, disable=self.silent)
            for i, sample in enumerate(tbar):
                image = sample['image']

                if self.cuda:
                    image = image.cuda()
                output = self.model(image)
                loss = self.criterion(output, image)
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

                train_loss += loss.item()
                # In futuro magari inseriremo delle metriche
                tbar.set_description('Epoch:%d, Train loss: %.3f' % (epoch, train_loss / (i + 1)))

            if self.save_intermediate and (epoch+1)%10 == 0:
                torch.save(self.model.state_dict(), os.path.join(save_path, f'{name}_{epoch}.pt'))

    def save_weights(self, filename):
        torch.save(self.model.state_dict(), os.path.join(filename)) #args.experiment_dir, filename))


def launch(data_path, epochs, batch_size, latent_dim, lambda_p, lambda_s, f, AE_type,
           save_intermediate=False, save_path='', use_cuda=True):
    trainer = Trainer_AE(latent_dim, lambda_p, lambda_s, f, data_path, AE_type, batch_size,
                      save_intermediate=save_intermediate, use_cuda=use_cuda)

    #summary(trainer.model, (1, 28, 28))

    start = time()
    trainer.train(epochs, save_path=save_path)
    tot_time = time() - start

    heatmaps, scores, gtmaps, labels = trainer.test()

    return heatmaps, scores, gtmaps, labels, tot_time



if __name__ == '__main__':
    def f(x):
        return 1-x

    data_path = 'test_data/mnist/0/2'
    info = data_path.split('/')
    dataset = info[-2]
    seed = info[-1]

    start = time()

    trainer = Trainer_AE(32, 25/(28*28), 1.2, f, data_path,
                      'shallow', 4)
    trainer.train(1000)
    print('TIME ELAPSED: ',time()-start)
    heatmaps, scores, gtmaps, labels = trainer.test()

    ret_path = os.path.join('results', dataset, seed)

    if not os.path.exists('results'):
        os.makedirs('results')
        os.makedirs(os.path.join('results', dataset))
        os.makedirs(ret_path)
    elif not os.path.exists(os.path.join('results', dataset)):
        os.makedirs(os.path.join('results', dataset))
        os.makedirs(ret_path)
    elif not os.path.exists(ret_path):
        os.makedirs(ret_path)

    np.save(open(os.path.join(ret_path, 'aexad_htmaps.npy'), 'wb'), heatmaps)
    np.save(open(os.path.join(ret_path, 'aexad_scores.npy'), 'wb'), scores)

    if not os.path.exists(os.path.join(ret_path, 'gt.npy')):
        np.save(open(os.path.join(ret_path, 'gt.npy'), 'wb'), gtmaps)

    if not os.path.exists(os.path.join(ret_path, 'labels.npy')):
        np.save(open(os.path.join(ret_path, 'labels.npy'), 'wb'), labels)