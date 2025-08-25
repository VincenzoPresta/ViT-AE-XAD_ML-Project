import torch
import torch.nn as nn
import numpy as np


class AEXAD_loss(nn.Module):

    def __init__(self, lambda_p=None, lambda_s=None, f=lambda x: torch.where(x >= 0.5, 0.0, 1.0), cuda=True):
        '''
        AEXAD loss
        :param lambda_p: float, anomalous pixels weight, if None it is inferred from the number of anomalous pixels, defaults to None
        :param lambda_s: float, anomalous samples weight, if None it is inferred from the number of anomalous samples, defaults to None
        :param f: func, function used to transform the anomalous pixels, defaults to lambda x: torch.where(x >= 0.5, 0.0, 1.0)
        :param cuda: bool, if True cuda is used, defaults to True
        '''
        super().__init__()
        self.lambda_p = lambda_p
        self.lambda_s = lambda_s
        self.f = f
        self.use_cuda = cuda

    def forward(self, input, target, gt, y):
        rec_n = (input - target) ** 2
        rec_o = (self.f(target) - input) ** 2

        if self.lambda_p is None:
            lambda_p = torch.reshape(np.prod(gt.shape[1:]) / torch.sum(gt, dim=(1, 2, 3)), (-1, 1))
            ones_v = torch.ones((gt.shape[0], np.prod(gt.shape[1:])))
            if self.use_cuda:
                lambda_p = lambda_p.cuda()
                ones_v = ones_v.cuda()
            lambda_p = ones_v * lambda_p
            lambda_p = torch.reshape(lambda_p, gt.shape)
            lambda_p = torch.where(gt == 1, lambda_p, 1 - gt)
        else:
            lambda_p = self.lambda_p
            if self.use_cuda:
                lambda_p = lambda_p.cuda()

        loss_vec = (1 - gt) * rec_n + lambda_p * gt * rec_o

        # Peso calcolato a livello di batch per momento: lambda_s calcolato qui
        loss = torch.sum(loss_vec, dim=(1, 2, 3))
        loss_n = torch.sum((1 - gt) * rec_n)
        loss_a = torch.sum(lambda_p * gt * rec_o)

        lambda_vec = torch.where(y == 1, self.lambda_s, 1.0)
        weighted_loss = torch.sum(loss * lambda_vec)
        return weighted_loss / torch.sum(lambda_vec), loss_n, loss_a # / torch.sum(lambda_vec)


class AEXAD_loss_norm(nn.Module):

    def __init__(self, lambda_p=None, lambda_s=None, f=lambda x: torch.where(x >= 0.5, 0.0, 1.0), cuda=True):
        '''
        AEXAD loss
        :param lambda_p: float, anomalous pixels weight, if None it is inferred from the number of anomalous pixels, defaults to None
        :param lambda_s: float, anomalous samples weight, if None it is inferred from the number of anomalous samples, defaults to None
        :param f: func, function used to transform the anomalous pixels, defaults to lambda x: torch.where(x >= 0.5, 0.0, 1.0)
        :param cuda: bool, if True cuda is used, defaults to True
        '''
        super().__init__()
        self.lambda_p = lambda_p
        self.lambda_s = lambda_s
        self.f = f
        self.use_cuda = cuda
        
        #fix: float' object has no attribute 'cuda'
        if lambda_p is not None:
            self.lambda_p = torch.tensor(lambda_p, dtype=torch.float32)
            if cuda:
                self.lambda_p = self.lambda_p.cuda()
        else:
            self.lambda_p = None

    def forward(self, rec_img, target, gt, y):
        '''
        :param rec_img: tensor, reconstructed image
        :param target: tensor, input image
        :param gt: tensor, ground truth image
        :param y: tensor, labels
        '''
        max_diff = (self.f(target) - target) ** 2
        
        print("[DEBUG loss] rec_img:", rec_img.shape)
        print("[DEBUG loss] target:", target.shape)
        print("[DEBUG loss] gt:", gt.shape)
        print("[DEBUG loss] y:", y.shape)
        
        '''# Allinea la maschera GT ai canali dell'immagine (per ViT RGB)
        if gt.shape[1] == 1 and rec_img.shape[1] == 3: #maschera grayscale, immagine rgb
            gt = gt.repeat(1, 3, 1, 1)'''
        
        rec_n = (rec_img - target) ** 2 / max_diff
        rec_o = (self.f(target) - rec_img) ** 2 / max_diff

        if self.lambda_p is None:
            lambda_p = torch.reshape(np.prod(gt.shape[1:]) / torch.sum(gt, dim=(1, 2, 3)), (-1, 1))
            ones_v = torch.ones((gt.shape[0], np.prod(gt.shape[1:])))
            if self.use_cuda:
                lambda_p = lambda_p.cuda()
                ones_v = ones_v.cuda()
            lambda_p = ones_v * lambda_p
            lambda_p = torch.reshape(lambda_p, gt.shape)
            lambda_p = torch.where(gt == 1, lambda_p, 1 - gt)
        else:
            lambda_p = self.lambda_p
            if self.use_cuda:
                lambda_p = lambda_p.cuda()

        loss_vec = (1 - gt) * rec_n + lambda_p * gt * rec_o

        # Peso calcolato a livello di batch per momento: lambda_s calcolato qui
        loss = torch.sum(loss_vec,
                         dim=(1, 2, 3))  # / torch.sum(lambda_p, dim=(1,2,3)) # Modifica 08/05/2025 aggiunta divisione
        loss_n = torch.sum((1 - gt) * rec_n) / torch.sum((1 - gt))  # Modifica 08/05/2025 prima: / rec_img.shape[0]
        loss_a = torch.sum(lambda_p * gt * rec_o) / torch.sum(
            lambda_p * gt)  # Modifica 08/05/2025 prima: / rec_img.shape[0]

        # lambda_vec = torch.where(y == 1, self.lambda_s, 1.0)
        # weighted_loss = torch.sum(loss * lambda_vec) / torch.sum(lambda_vec)
        weighted_loss = torch.mean(loss)
        return weighted_loss, loss_n, loss_a  # / torch.sum(lambda_vec)

class AEXAD_loss_weighted(nn.Module):

    def __init__(self, lambda_p, lambda_s, norm, f, cuda):
        super().__init__()
        self.lambda_p = lambda_p
        self.lambda_s = lambda_s
        self.norm = norm           # norm = a_max * (n_normali / sum(max_area / tot_pixel_anom))
        self.f = f
        self.use_cuda = cuda

    def forward(self, input, target, gt, y):
        rec_n = (input - target) ** 2
        rec_o = (self.f(target) - input) ** 2

        if self.lambda_p is None:
            lambda_p_v = torch.reshape(np.prod(gt.shape[1:]) / torch.sum(gt, dim=(1, 2, 3)), (-1, 1))
            ones_v = torch.ones((gt.shape[0], np.prod(gt.shape[1:])))
            if self.use_cuda:
                lambda_p_v = lambda_p_v.cuda()
                ones_v = ones_v.cuda()
            lambda_p = ones_v * lambda_p_v
            lambda_p = torch.reshape(lambda_p, gt.shape)
            lambda_p = torch.where(gt == 1, lambda_p, 1 - gt)
        else:
            lambda_p = torch.full((gt.shape[0], np.prod(gt.shape[1:])), fill_value=self.lambda_p)
            if self.use_cuda:
                lambda_p = lambda_p.cuda()
            lambda_p = torch.reshape(lambda_p, gt.shape)
            lambda_p = torch.where(gt == 1, lambda_p, 1 - gt)


        loss_vec = (1 - gt) * rec_n + lambda_p * gt * rec_o

        # Peso calcolato a livello di batch per momento: lambda_s calcolato qui
        #print(loss_vec.shape)
        loss = torch.sum(loss_vec, dim=(1, 2, 3)) / torch.sum(lambda_p, dim=(1, 2, 3))
        #lambda_vec = torch.Tensor(np.where(y==1, self.lambda_s, 1.0))
        areas_a = torch.sum(gt, dim=(1, 2, 3))
        lambda_vec = self.norm / torch.where(y == 1, areas_a, torch.full_like(areas_a, fill_value=self.norm, dtype=torch.float32))  # If the sample is normal the resulting weigth is 1.

        weighted_loss = torch.sum(loss * lambda_vec)
        return weighted_loss / torch.sum(lambda_vec)

class AEXAD_loss_norm_vgg(nn.Module):

    def __init__(self, mean, std, lambda_p=None, lambda_s=None, f=lambda x: torch.where(x >= 0.5, 0.0, 1.0), cuda=True):
        '''
        AEXAD loss
        :param lambda_p: float, anomalous pixels weight, if None it is inferred from the number of anomalous pixels, defaults to None
        :param lambda_s: float, anomalous samples weight, if None it is inferred from the number of anomalous samples, defaults to None
        :param f: func, function used to transform the anomalous pixels, defaults to lambda x: torch.where(x >= 0.5, 0.0, 1.0)
        :param cuda: bool, if True cuda is used, defaults to True
        '''
        super().__init__()
        self.lambda_p = lambda_p
        self.lambda_s = lambda_s
        self.f = f
        self.use_cuda = cuda
        self.mean = mean
        self.std = std

    def forward(self, rec_img, target, gt, y):
        '''
        :param rec_img: tensor, reconstructed image
        :param target: tensor, input image
        :param gt: tensor, ground truth image
        :param y: tensor, labels
        '''
        #mean = torch.from_numpy(np.full((target.shape[0], target.shape[1], target.shape[2]*target.shape[3]), fill_value=self.mean[np.newaxis, :, np.newaxis])).cuda()
        #std = torch.from_numpy(np.full((target.shape[0], target.shape[1], target.shape[2]*target.shape[3]), fill_value=self.std[np.newaxis, :, np.newaxis])).cuda()
        mean = torch.from_numpy(self.mean[np.newaxis, :, np.newaxis, np.newaxis]).cuda()
        std = torch.from_numpy(self.std[np.newaxis, :,  np.newaxis, np.newaxis]).cuda()
        #d_target = (target.reshape((target.shape[0], target.shape[1], -1)) * std + mean).reshape(target.shape)
        d_target = target * std + mean
        max_diff = (self.f(d_target) - d_target) ** 2
        rec_n = (rec_img - d_target) ** 2 / max_diff
        rec_o = (self.f(d_target) - rec_img) ** 2 / max_diff

        if self.lambda_p is None:
            lambda_p = torch.reshape(np.prod(gt.shape[1:]) / torch.sum(gt, dim=(1, 2, 3)), (-1, 1))
            ones_v = torch.ones((gt.shape[0], np.prod(gt.shape[1:])))
            if self.use_cuda:
                lambda_p = lambda_p.cuda()
                ones_v = ones_v.cuda()
            lambda_p = ones_v * lambda_p
            lambda_p = torch.reshape(lambda_p, gt.shape)
            lambda_p = torch.where(gt == 1, lambda_p, 1 - gt)
        else:
            lambda_p = self.lambda_p
            if self.use_cuda:
                lambda_p = lambda_p.cuda()

        loss_vec = (1 - gt) * rec_n + lambda_p * gt * rec_o

        # Peso calcolato a livello di batch per momento: lambda_s calcolato qui
        loss = torch.sum(loss_vec, dim=(1, 2, 3)) #/ torch.sum(lambda_p, dim=(1,2,3)) # Modifica 08/05/2025 aggiunta divisione
        loss_n = torch.sum((1 - gt) * rec_n) / torch.sum((1 - gt))                   # Modifica 08/05/2025 prima: / rec_img.shape[0]
        loss_a = torch.sum(lambda_p * gt * rec_o) / torch.sum(lambda_p * gt)         # Modifica 08/05/2025 prima: / rec_img.shape[0]

        #lambda_vec = torch.where(y == 1, self.lambda_s, 1.0)
        #weighted_loss = torch.sum(loss * lambda_vec) / torch.sum(lambda_vec)
        weighted_loss = torch.mean(loss)
        return weighted_loss, loss_n, loss_a # / torch.sum(lambda_vec)

class MSE_loss_vgg(nn.Module):
    def __init__(self, mean, std, cuda=True):
        '''
        AEXAD loss
        :param lambda_p: float, anomalous pixels weight, if None it is inferred from the number of anomalous pixels, defaults to None
        :param lambda_s: float, anomalous samples weight, if None it is inferred from the number of anomalous samples, defaults to None
        :param f: func, function used to transform the anomalous pixels, defaults to lambda x: torch.where(x >= 0.5, 0.0, 1.0)
        :param cuda: bool, if True cuda is used, defaults to True
        '''
        super().__init__()
        self.use_cuda = cuda
        self.mean = mean
        self.std = std
        self.mse = nn.MSELoss()

    def forward(self, rec_img, target):
        '''
        :param rec_img: tensor, reconstructed image
        :param target: tensor, input image
        :param gt: tensor, ground truth image
        :param y: tensor, labels
        '''
        mean = torch.from_numpy(self.mean[np.newaxis, :, np.newaxis, np.newaxis]).cuda()
        std = torch.from_numpy(self.std[np.newaxis, :,  np.newaxis, np.newaxis]).cuda()
        d_target = target * std + mean
        d_target = d_target.float()

        return self.mse(rec_img, d_target)