import os

import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from AE_architectures import Shallow_Autoencoder, Deep_Autoencoder, Conv_Autoencoder, PCA_Autoencoder, \
    Conv_Deep_Autoencoder, Conv_Autoencoder_f2, Conv_Deep_Autoencoder_v2
from datasets.custom import CustomAD
from datasets.mvtec import MvtecAD
from datasets.augmented import AugmentedAD
from loss import AEXAD_loss, AEXAD_loss_weighted

class Trainer:
    def __init__(self, latent_dim, lambda_p, lambda_s, f, path, AE_type, batch_size=None, silent=False, use_cuda=True,
                 loss='aexad', save_intermediate=False, dataset='mnist'):
        '''
        :param latent_dim:
        :param lambda_p: float, anomalous pixel weight, if none the value is inferred from the dataset
        :param lambda_s: float, anomalous samples weight, if none the value is inferred from the dataset
        :param f: func, mapping function for anomalous pixels
        :param path: str path in which checkpoints are saved
        :param AE_type: str Type of architecture considered. Possible values are: shallow, deep, conv, conv_deep, conv_f2, pca
        :param batch_size: int number of samples composing a batch, defaults to None
        :param silent: bool, if True, deactivate progress bar, defaults to False
        :param use_cuda: bool, if True the network uses gpu, defaults to True
        :param loss: str, loss to use, possible values are aexad and mse, defaults to aexad
        :param save_intermediate: bool, if True save a checkpoint of the model every 100 epochs, defaults to False
        :param dataset: str, dataset to use, default to mnist, for custom dataset type the dataset name
        '''
        self.silent = silent

        if dataset == 'mnist' or dataset == 'fmnist':
            self.train_loader = DataLoader(CustomAD(path, train=True), batch_size=batch_size, shuffle=True)
            self.test_loader = DataLoader(CustomAD(path, train=False), batch_size=batch_size, shuffle=False)
        elif dataset == 'mvtec':
            self.train_loader = DataLoader(MvtecAD(path, train=True), batch_size=batch_size, shuffle=True)
            self.test_loader = DataLoader(MvtecAD(path, train=False), batch_size=batch_size, shuffle=False)
        else:
            print(dataset)
            ds = AugmentedAD(path, train=True)
            weights = np.where(ds.labels == 1, 0.6, 0.4)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=len(ds.labels))
            self.train_loader = DataLoader(ds, batch_size=batch_size, sampler=sampler)
            self.test_loader = DataLoader(AugmentedAD(path, train=False), batch_size=batch_size, shuffle=False)

        self.save_intermediate = save_intermediate

        if lambda_s is None:
            lambda_s = len(self.train_loader.dataset) / np.sum(self.train_loader.dataset.labels)
            print(lambda_s)

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

        elif AE_type == 'conv_deep':
            self.model = Conv_Deep_Autoencoder(self.train_loader.dataset.dim)

        elif AE_type == 'conv_deep_v2':
            self.model = Conv_Deep_Autoencoder_v2(self.train_loader.dataset.dim)

        elif AE_type == 'conv_f2':
            self.model = Conv_Autoencoder_f2(self.train_loader.dataset.dim)

        elif AE_type == 'pca':
            self.model = PCA_Autoencoder(np.prod(self.train_loader.dataset.dim), np.prod(self.train_loader.dataset.dim),
                                         latent_dim)
        else:
            raise Exception('Model not yet implemented')

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.loss = loss
        if loss == 'aexad':
            self.criterion = AEXAD_loss(lambda_p, lambda_s, f, self.cuda)
        elif loss == 'aexad_norm':
            self.criterion = AEXAD_loss_weighted(lambda_p, lambda_s, f, self.cuda)
        elif loss == 'mse':
            self.criterion = nn.MSELoss()

        if self.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

    def reconstruct(self):
        '''
        Reconstruct the images of the test data loader
        '''
        self.model.eval()
        tbar = tqdm(self.test_loader, disable=self.silent)
        shape = [0]
        shape.extend(self.test_loader.dataset.images.shape[1:])
        rec_images = []
        for i, sample in enumerate(tbar):
            image, label, gtmap = sample['image'], sample['label'], sample['gt_label']
            if self.cuda:
                image = image.cuda()
            output = self.model(image).detach().cpu().numpy()
            rec_images.extend(output)
        rec_images = np.array(rec_images)
        rec_images = rec_images.swapaxes(1, 2).swapaxes(2, 3)
        return rec_images


    def test(self):
        '''
        Test the model on the test set provided by the test data loader
        '''
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
            if self.cuda:
                image = image.cuda()
            output = self.model(image).detach().cpu().numpy()
            image = image.cpu().numpy()
            heatmap = np.abs(image-output) #((image-output) ** 2)#.numpy()
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


    def train(self, epochs, save_path='.'):
        '''
        Trains the model on the train set provided by the train data loader
        :param epochs: int, number of epochs
        :param save_path: str, path used for saving the model, defaults to '.'
        '''
        if isinstance(self.model, Conv_Autoencoder):
            name = 'model_conv'
        elif isinstance(self.model, Conv_Autoencoder_f2):
            name = 'model_conv_f2'
        elif isinstance(self.model, Deep_Autoencoder):
            name = 'model_deep'
        elif isinstance(self.model, Shallow_Autoencoder):
            name = 'model'
        elif isinstance(self.model, Conv_Deep_Autoencoder):
            name = 'model_conv_deep'
        elif isinstance(self.model, Conv_Deep_Autoencoder_v2):
            name = 'model_conv_deep_v2'

        self.model.train()
        for epoch in range(epochs):
            train_loss = 0.0
            train_loss_n = 0.0
            train_loss_a = 0.0
            na = 0
            nn = 0
            ns = 0
            tbar = tqdm(self.train_loader, disable=self.silent)
            for i, sample in enumerate(tbar):
                image, label, gt_label = sample['image'], sample['label'], sample['gt_label']

                if self.cuda:
                    image = image.cuda()
                    gt_label = gt_label.cuda()
                    label = label.cuda()
                output = self.model(image)
                if self.loss == 'mse':
                    loss = self.criterion(output, image)
                else:
                    loss, loss_n, loss_a = self.criterion(output, image, gt_label, label)
                    train_loss_n += loss_n.item()
                    train_loss_a += loss_a.item()
                    na += label.sum()
                    nn += image.shape[0] - na
                ns += 1
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

                train_loss += loss.item()
                # In futuro magari inseriremo delle metriche

                if self.loss == 'mse':
                    tbar.set_description('Epoch:%d, Train loss: %.3f' % (epoch, train_loss / ns))
                else:
                    print('ok')
                    tbar.set_description('Epoch:%d, Train loss: %.3f, Normal loss: %.3f, Anom loss: %3f' % (epoch, train_loss / ns, train_loss_n / nn, train_loss_a / na))

            if self.save_intermediate and (epoch+1)%100 == 0:
                torch.save(self.model.state_dict(), os.path.join(save_path, f'{self.loss}_{name}_{epoch+1}.pt'))

    def save_weights(self, filename):
        torch.save(self.model.state_dict(), os.path.join(filename)) #args.experiment_dir, filename))
