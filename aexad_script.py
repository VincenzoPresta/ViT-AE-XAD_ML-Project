import os

import matplotlib.pyplot as plt
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from time import time

from AE_architectures import Shallow_Autoencoder, Deep_Autoencoder, Conv_Autoencoder, PCA_Autoencoder, \
    Conv_Deep_Autoencoder, Conv_Deep_Autoencoder_v2, VGG_CNN_mask, ResNet_CNN_mask, \
    ViT_CNN_mask, Conv_Deep_Autoencoder_v2_Attn, ViT_CNN_Attn, ResNet_CNN_Attn
from aexad_dataloaders.dataset import CustomAD
from aexad_dataloaders.mvtec_dataset import MvtecAD
from aexad_dataloaders.custom_VGG import CustomVGGAD
from augmented import AugmentedAD
from loss import AEXAD_loss, AEXAD_loss_weighted, AEXAD_loss_norm, AEXAD_loss_norm_vgg, MSE_loss_vgg


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

        if dataset == 'mnist' or dataset == 'fmnist' or dataset == 'tf_ds':
            if AE_type == 'vgg_cnn' or AE_type == 'conv_deep_v2_attn':
                print(AE_type)
                ds = CustomVGGAD(path, train=True)
                weights = np.where(ds.labels == 1, 0.8, 0.2)
                sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=len(ds.labels))
                self.train_loader = DataLoader(ds, batch_size=batch_size, sampler=sampler)
                self.test_loader = DataLoader(CustomVGGAD(path, train=False), batch_size=1, shuffle=False)
            else:
                print('no_vgg')
                ds = CustomAD(path, train=True)
                weights = np.where(ds.labels == 1, 0.6, 0.4)
                sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=len(ds.labels))
                self.train_loader = DataLoader(ds, batch_size=batch_size, sampler=sampler)
                #self.train_loader = DataLoader(CustomAD(path, train=True), batch_size=batch_size, shuffle=True)
                self.test_loader = DataLoader(CustomAD(path, train=False), batch_size=batch_size, shuffle=False)
        elif dataset == 'mvtec':
            print(dataset)
            ds = MvtecAD(path, train=True)
            #weights = np.where(ds.labels == 1, 0.6, 0.4)
            #sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=len(ds.labels))
            self.train_loader = DataLoader(ds, batch_size=batch_size)#, sampler=sampler)
            #self.test_loader = DataLoader(AugmentedAD(path, train=False), batch_size=batch_size, shuffle=False)
            self.test_loader = DataLoader(MvtecAD(path, train=False), batch_size=batch_size, shuffle=False)
        else:
            print('E ', dataset)
            ds = MvtecAD(path, train=True)
            weights = np.where(ds.labels == 1, 0.6, 0.4)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=len(ds.labels))
            self.train_loader = DataLoader(ds, batch_size=batch_size, sampler=sampler)
            #self.test_loader = DataLoader(AugmentedAD(path, train=False), batch_size=batch_size, shuffle=False)
            self.test_loader = DataLoader(MvtecAD(path, train=False), batch_size=batch_size, shuffle=False)

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

        elif AE_type == 'conv_deep_v2_attn':
            self.model = Conv_Deep_Autoencoder_v2_Attn(self.train_loader.dataset.dim)

        elif AE_type == 'vgg_cnn':
            assert isinstance(self.train_loader.dataset, CustomVGGAD)
            self.model = ResNet_CNN_Attn(self.train_loader.dataset.dim)

        elif AE_type == 'pca':
            self.model = PCA_Autoencoder(np.prod(self.train_loader.dataset.dim), np.prod(self.train_loader.dataset.dim),
                                         latent_dim)
        
        #Nuovo: ViT    
        elif AE_type == 'vit':
             # Nota importante:
            # A differenza degli altri modelli, ViT usa come backbone vit_b_16 pre-addestrato su ImageNet.
            # Questo backbone accetta SOLO input RGB (3 canali, 224x224).
            # Non possiamo affidarci a self.train_loader.dataset.dim perché può contenere shape "storte"
            # (es. (224,1,224)), dovute a permute/repeat nel preprocessing.
            # Per questo motivo forziamo manualmente la shape a (3,224,224).
            self.model = ViT_CNN_Attn((3, 224, 224))

                        
        else:
            raise Exception('Model not yet implemented')

        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        #self.optimizer = torch.optim.Adam([{'params': self.model.encoder.parameters()},
        #                                  {'params': self.model.dec1.parameters()},
        #                                  {'params': self.model.dec2.parameters()},
        #                                  {'params': self.model.dec3.parameters()},
        #                                  {'params': self.model.decoder_final.parameters()},
        #                                  {'params': self.model.encoder_pre.parameters(), 'lr': 1e-5}
        #                                  ], lr=1e-3, weight_decay=1e-4)
        
        # ResNet
        '''self.optimizer = torch.optim.Adam([{'params': self.model.encoder.parameters()},
                                           {'params': self.model.dec1.parameters()},
                                           {'params': self.model.dec2.parameters()},
                                           {'params': self.model.dec3.parameters()},
                                           {'params': self.model.decoder_final.parameters()},
                                           {'params': self.model.conv1.parameters(), 'lr': 1e-5},
                                           {'params': self.model.bn1.parameters(), 'lr': 1e-5},
                                           {'params': self.model.relu1.parameters(), 'lr': 1e-5},
                                           {'params': self.model.maxpool1.parameters(), 'lr': 1e-5},
                                           {'params': self.model.layer1.parameters(), 'lr': 1e-5},
                                           {'params': self.model.layer2.parameters(), 'lr': 1e-5},
                                           ], lr=1e-3, weight_decay=1e-4)'''
                                           
        #ViTCNN
        self.optimizer = torch.optim.Adam([
                                         {'params': self.model.decoder1.parameters()},
                                         {'params': self.model.decoder2.parameters()},
                                         {'params': self.model.encoder1.parameters()},
                                         {'params': self.model.encoder.parameters(), 'lr': 1e-5}
                                         ], lr=1e-3, weight_decay=1e-4) 
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda ep: 0.985 ** ep)

        self.loss = loss
        if loss == 'aexad':
            if isinstance(self.train_loader.dataset, CustomVGGAD):
                self.criterion = AEXAD_loss_norm_vgg(self.train_loader.dataset.mean,
                                 self.train_loader.dataset.std, lambda_p, lambda_s, f, self.cuda)
            else:
                self.criterion = AEXAD_loss_norm(lambda_p, lambda_s, f, self.cuda)
            #self.criterion = AEXAD_loss(lambda_p, lambda_s, f, self.cuda)
        elif loss == 'aexad_norm':
            self.criterion = AEXAD_loss_weighted(lambda_p, lambda_s, f, self.cuda)
        elif loss == 'mse':
            if isinstance(self.train_loader.dataset, CustomVGGAD):
                self.criterion = MSE_loss_vgg(self.train_loader.dataset.mean, self.train_loader.dataset.std)
            else:
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
        if isinstance(self.train_loader.dataset, CustomVGGAD):
            rec_images = rec_images * self.train_loader.dataset.std + self.train_loader.dataset.std
        return rec_images


    def test(self):
        '''
        Test the model on the test set provided by the test data loader
        '''
        tbar = tqdm(self.test_loader, disable=self.silent)
        shape = [0]
        shape.extend(self.test_loader.dataset.images.shape[1:])
        heatmaps = []
        scores = []
        gtmaps = []
        labels = []

        self.model.eval()
        with torch.no_grad():
            for i, sample in enumerate(tbar):
                image, label, gtmap = sample['image'], sample['label'], sample['gt_label']
                if self.cuda:
                    image = image.cuda()
                    output = self.model(image).detach().cpu().numpy()
                image = image.cpu().numpy()
                if isinstance(self.train_loader.dataset, CustomVGGAD):
                    print('Testing on vgg')
                    mean = self.test_loader.dataset.mean[np.newaxis, :, np.newaxis, np.newaxis]
                    std = self.test_loader.dataset.std[np.newaxis, :, np.newaxis, np.newaxis]
                    #image = (image.reshape((image.shape[0], image.shape[1], -1)) * std + mean).reshape(image.shape)
                    image = image * std + mean
                heatmap = ((image-output) ** 2)#.numpy() np.abs(image-output)
                score = heatmap.reshape((image.shape[0], -1)).mean(axis=-1)
                heatmaps.extend(heatmap)
                scores.extend(score)
                gtmaps.extend(gtmap.detach().numpy())
                labels.extend(label.detach().numpy())
                plt.figure()
                plt.subplot(1, 3, 1)
                plt.imshow(image[0].swapaxes(0, 1).swapaxes(1, 2))
                plt.subplot(1, 3, 2)
                plt.imshow(output[0].swapaxes(0, 1).swapaxes(1, 2))
                plt.subplot(1, 3, 3)
                plt.imshow(gtmap[0].numpy().swapaxes(0, 1).swapaxes(1, 2))
                plt.savefig(f'test_{i}_{self.loss}.jpg')
                plt.close('all')
            scores = np.array(scores)
            heatmaps = np.array(heatmaps)
            gtmaps = np.array(gtmaps)
            labels = np.array(labels)
        return heatmaps, scores, gtmaps, labels


    def stop_fe_training(self):
        assert isinstance(self.model, VGG_CNN_mask)

        for param in self.model.encoder_pre.parameters():
            param.requires_grad = False


    def train(self, epochs, save_path='.'):
        '''
        Trains the model on the train set provided by the train data loader
        :param epochs: int, number of epochs
        :param save_path: str, path used for saving the model, defaults to '.'
        '''
        if isinstance(self.model, Conv_Autoencoder):
            name = 'model_conv'
        elif isinstance(self.model, Deep_Autoencoder):
            name = 'model_deep'
        elif isinstance(self.model, Shallow_Autoencoder):
            name = 'model'
        elif isinstance(self.model, Conv_Deep_Autoencoder):
            name = 'model_conv_deep'
        elif isinstance(self.model, Conv_Deep_Autoencoder_v2):
            name = 'model_conv_deep_v2_attn'
        elif isinstance(self.model, Conv_Deep_Autoencoder_v2_Attn):
            name = 'model_conv_deep_v2'
        elif isinstance(self.model, ResNet_CNN_Attn):
            name = 'model_vgg_cnn'
        
        #nuovo: ViT
        elif isinstance(self.model, ViT_CNN_Attn):
            print("[DEBUG dataset dim]", self.train_loader.dataset.dim)
            name = 'model_vit_cnn'
                    
        print("DEBUG:", name)

        fe_untrain = False
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
                # Forza sempre (N, C, H, W)
                if image.ndim == 4:
                    # Caso ideale: già (N,1,224,224) o (N,3,224,224)
                    if image.shape[1] in [1, 3] and image.shape[2] == 224 and image.shape[3] == 224:
                        pass  # già a posto
                    # Caso sbagliato: (N,224,224,1) -> (N,1,224,224)
                    elif image.shape[-1] in [1, 3]:
                        image = image.permute(0, 3, 1, 2)
                    # Caso sbagliato: (N,224,1,224) -> (N,1,224,224)
                    elif image.shape[2] in [1, 3]:
                        image = image.permute(0, 2, 1, 3)

                print("[DEBUG after permute]", image.shape)

                if self.cuda:
                    image = image.cuda()
                    gt_label = gt_label.cuda()
                    label = label.cuda()
                    
                #Check per ViT: caso MNIST/FMNIST (grayscale → RGB)
                if isinstance(self.model, ViT_CNN_Attn) and image.ndim == 4 and image.shape[1] == 1:
                    image = image.repeat(1, 3, 1, 1)           
                    
                output = self.model(image)
                
                print("[DEBUG before loss] output:", output.shape, "image:", image.shape)
                
                if self.loss == 'mse':
                    loss = self.criterion(output, image)
                else:
                    loss, loss_n, loss_a = self.criterion(output, image, gt_label, label)
                    na += label.sum()
                    nn += image.shape[0] - na
                ns += 1
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

                train_loss += loss.item()
                if not self.loss == 'mse':
                    train_loss_n += loss_n.item()
                    train_loss_a += loss_a.item()
                # In futuro magari inseriremo delle metriche

                #plt.figure()
                #plt.subplot(1, 2, 1)
                #plt.imshow(image.detach().cpu().numpy()[0].swapaxes(0, 1).swapaxes(1, 2))
                #plt.subplot(1, 2, 2)
                #plt.imshow(output.detach().cpu().numpy()[0].swapaxes(0, 1).swapaxes(1, 2))
                #plt.savefig(f'train_{i}_{epoch+1}.jpg')
                #plt.close('all')

                if self.loss == 'mse':
                    tbar.set_description('Epoch:%d, Train loss: %.3f' % (epoch, train_loss / ns))
                else:
                    tbar.set_description('Epoch:%d, Train loss: %.3f, Normal loss: %.3f, Anom loss: %3f' % (epoch, train_loss / ns, loss_n, loss_a))

                #if isinstance(self.model, VGG_CNN_mask) and epoch>0.1*epochs:
                #    self.stop_fe_training()

            #if self.save_intermediate and not fe_untrain and (epoch+1) % 100 == 0:
            #    torch.save(self.model.state_dict(), os.path.join(save_path, f'{self.loss}_{name}_{epoch+1}.pt'))
            #    fe_untrain = True

            self.scheduler.step()

        torch.save(self.model.state_dict(), os.path.join(save_path, f'{self.loss}_{name}.pt'))

    def save_weights(self, filename):
        torch.save(self.model.state_dict(), os.path.join(filename)) #args.experiment_dir, filename))


def launch(data_path, epochs, batch_size, latent_dim, lambda_p, lambda_s, f, AE_type, loss='aexad',
           save_intermediate=False, save_path='', use_cuda=True, dataset='mnist'):
    trainer = Trainer(latent_dim, lambda_p, lambda_s, f, data_path, AE_type, batch_size, loss=loss,
                      save_intermediate=save_intermediate, use_cuda=use_cuda, dataset=dataset)

    #summary(trainer.model, (3, 448, 448))

    start = time()
    trainer.train(epochs, save_path=save_path)
    tot_time = time() - start

    heatmaps, scores, gtmaps, labels = trainer.test()
    #torch.save(trainer.model.state_dict(), os.path.join(save_path, f'{loss}_conv_deep_v2.pt'))

    return heatmaps, scores, gtmaps, labels, tot_time
