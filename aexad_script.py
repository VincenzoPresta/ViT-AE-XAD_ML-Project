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
from loss import AEXAD_loss, AEXAD_loss_weighted, AEXAD_loss_norm, AEXAD_loss_norm_vgg, MSE_loss_vgg, AEXAD_loss_ViT


class Trainer:
    def __init__(self, latent_dim, lambda_p, lambda_s, f, path, AE_type,batch_size=None, silent=False, use_cuda=True,
                 loss='aexad', save_intermediate=False, dataset='mnist', save_path="." ):
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
        self.save_path = save_path

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
        
        
#-------OPTIMIZERS------------------------------

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
                                           
        # ViTCNN: encoder ViT + decoder ResNet AE-XAD
        self.optimizer = torch.optim.Adam([
            {'params': self.model.encoder.parameters(), 'lr': 1e-5},   # ViT_Encoder: conv_proj, encoder_vit, to_64, up_to_28
            {'params': self.model.decoder.parameters(), 'lr': 1e-3},   # dec1, dec2, dec3, decoder_final
        ], lr=1e-3, weight_decay=1e-5)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda ep: 0.985 ** ep)

#--------LOSS--------------

        self.loss = loss
        if loss == 'aexad':
            if isinstance(self.train_loader.dataset, CustomVGGAD):
                self.criterion = AEXAD_loss_norm_vgg(self.train_loader.dataset.mean,
                                 self.train_loader.dataset.std, lambda_p, lambda_s, f, self.cuda)
            else:
                if isinstance(self.model, ViT_CNN_Attn):
                    self.criterion = AEXAD_loss_ViT(lambda_p, lambda_s, f, self.cuda)
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
            
#---------------------------

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
        heatmaps, scores, gtmaps, labels = [], [], [], []

        self.model.eval()
        
        os.makedirs(self.save_path, exist_ok=True)
        results_dir = os.path.join(self.save_path, "test_images")
        os.makedirs(results_dir, exist_ok=True)
                
        with torch.no_grad():
            for i, sample in enumerate(tbar):
                image, label, gtmap = sample['image'], sample['label'], sample['gt_label']

                # PATCH: correggi shape se sono storte (N,H,C,W) → (N,C,H,W)
                if image.ndim == 4 and image.shape[2] in [1,3] and image.shape[1] not in [1,3]:
                    image = image.permute(0, 2, 1, 3)
                if gtmap.ndim == 4 and gtmap.shape[2] == 1 and gtmap.shape[1] != 1:
                    gtmap = gtmap.permute(0, 2, 1, 3)

                # Se ViT è grayscale → duplica canali
                if isinstance(self.model, ViT_CNN_Attn) and image.shape[1] == 1:
                    image = image.repeat(1, 3, 1, 1)

                if self.cuda:
                    image = image.cuda()
                    output = self.model(image).detach().cpu().numpy()
                image = image.cpu().numpy()

                heatmap = ((image - output) ** 2)  # (B, C, H, W)

                # Somma sui canali → (B, H, W)
                heatmap = heatmap.sum(axis=1)
                
                heatmap = heatmap / (heatmap.max(axis=(1,2), keepdims=True) + 1e-8) # NORMALIZZAZIONE PER IMMAGINE – obbligatoria per AE-XAD Arrays
                score = heatmap.reshape((image.shape[0], -1)).mean(axis=-1)

                heatmaps.extend(heatmap)

                
                scores.extend(score)
                gtmaps.extend(gtmap.detach().numpy())
                labels.extend(label.detach().numpy())

                plt.figure(figsize=(14,4))

                plt.subplot(1, 4, 1)
                plt.imshow(image[0].swapaxes(0, 1).swapaxes(1, 2))
                plt.title("Input"); plt.axis("off")

                plt.subplot(1, 4, 2)
                plt.imshow(output[0].swapaxes(0, 1).swapaxes(1, 2))
                plt.title("Ricostruzione"); plt.axis("off")

                plt.subplot(1, 4, 3)
                plt.imshow(heatmap[0], cmap="hot")   # somma canali per avere 2D -> modificato così perche è gia 2d
                plt.title("Heatmap"); plt.axis("off")

                plt.subplot(1, 4, 4)
                plt.imshow(gtmap[0].cpu().numpy().squeeze(), cmap="gray")
                plt.title("GT mask"); plt.axis("off")

                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f"test_{i}_{self.loss}.jpg"))                
                plt.close("all")

            return np.array(heatmaps), np.array(scores), np.array(gtmaps), np.array(labels)



    def stop_fe_training(self):
        assert isinstance(self.model, VGG_CNN_mask)

        for param in self.model.encoder_pre.parameters():
            param.requires_grad = False


    def train(self, epochs=200, save_path='.'):
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
            name = 'model_vit_cnn'
                    
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
                
                print("[DEBUG train] image:", image.shape,
                    "label:", label.shape,
                    "gt_label:", gt_label.shape)
                
                # PATCH: forza sempre (N,C,H,W)
                if image.ndim == 4 and image.shape[2] in [1,3] and image.shape[1] not in [1,3]:
                    # Caso (N,H,C,W) -> (N,C,H,W)
                    image = image.permute(0, 2, 1, 3)

                if gt_label.ndim == 4 and gt_label.shape[2] == 1 and gt_label.shape[1] != 1:
                    # Caso (N,H,C,W) -> (N,C,H,W)
                    gt_label = gt_label.permute(0, 2, 1, 3)

                # Se ViT è grayscale → duplica a 3 canali
                if isinstance(self.model, ViT_CNN_Attn) and image.ndim == 4 and image.shape[1] == 1:
                    image = image.repeat(1, 3, 1, 1)

                print("[DEBUG train FIXED] image:", image.shape,
                    "gt_label:", gt_label.shape)
                
                print("[DEBUG values] image min:", image.min().item(),
                    "max:", image.max().item(),
                    "mean:", image.mean().item())


                if self.cuda:
                    image = image.cuda()
                    gt_label = gt_label.cuda()
                    label = label.cuda()
                    
                output = self.model(image)
                                
                if self.loss == 'mse':
                    # ----------------------- L1 + MSE -----------------------
                    loss_l1  = torch.nn.functional.l1_loss(output, image)
                    loss_mse = torch.nn.functional.mse_loss(output, image)
                    # bilanciamento ottimale per ricostruzione
                    loss = 0.7 * loss_l1 + 0.3 * loss_mse
                    # ---------------------------------------------------------
                else:
                    loss, loss_n, loss_a = self.criterion(output, image, gt_label, label)
                    na += label.sum()
                    nn += image.shape[0] - na
                
                ns += 1
                train_loss += loss.item()
                
                if not self.loss == 'mse':
                    train_loss_n += loss_n.item()
                    train_loss_a += loss_a.item()

                if self.loss == 'mse':
                    # solo AE puro
                    tbar.set_description('Epoch:%d, Train loss: %.3f' % (epoch, train_loss / ns))
                else:
                    # AE-XAD supervisionato
                    tbar.set_description('Epoch:%d, Train loss: %.3f, Normal loss: %.3f, Anom loss: %3f' % 
                                        (epoch, train_loss / ns, loss_n, loss_a))
                    
                # ===== BACKPROPAGATION =====
                self.optimizer.zero_grad()
                loss.backward()
                
                # ===== GRADIENT CLIPPING (ViT stabilità) =====
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                # ============================

            self.scheduler.step()

        torch.save(self.model.state_dict(), os.path.join(save_path, f'{self.loss}_{name}.pt'))

    def save_weights(self, filename):
        torch.save(self.model.state_dict(), os.path.join(filename)) #args.experiment_dir, filename))


def launch(data_path, batch_size, latent_dim, lambda_p, lambda_s, f, AE_type, epochs=200, loss='aexad',
           save_intermediate=False, save_path='', use_cuda=True, dataset='mnist'):
    
    trainer = Trainer(latent_dim, lambda_p, lambda_s, f, data_path, AE_type, batch_size, loss=loss,
                      save_intermediate=save_intermediate, use_cuda=use_cuda, dataset=dataset, save_path=save_path)

    #summary(trainer.model, (3, 448, 448))

    start = time()
    trainer.train(epochs=epochs, save_path=save_path)
    tot_time = time() - start

    heatmaps, scores, gtmaps, labels = trainer.test()
    #torch.save(trainer.model.state_dict(), os.path.join(save_path, f'{loss}_conv_deep_v2.pt'))

    return heatmaps, scores, gtmaps, labels, tot_time
