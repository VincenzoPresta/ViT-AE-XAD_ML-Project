import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



def gaussian_window(window_size, sigma):
    gauss = torch.arange(window_size).float() - window_size // 2
    gauss = torch.exp(-(gauss ** 2) / (2 * sigma ** 2))
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D = _1D.mm(_1D.t()).float()
    window = _2D.unsqueeze(0).unsqueeze(0)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# -----------------------------------------
# SSIM (versione PyTorch stabile)
# -----------------------------------------
def ssim(img1, img2, window_size=11, channel=3, size_average=True):
    device = img1.device

    window = create_window(window_size, channel).to(device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(dim=(1,2,3))



# --------------------------------------------------
# LOSS 3-TERMINE per ViT-AE-XAD
# --------------------------------------------------
class AEXAD_loss_ViT_SSIM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        l1 = F.l1_loss(output, target)
        l2 = F.mse_loss(output, target)
        ssim_val = 1 - ssim(output, target, channel=output.shape[1])

        loss = 0.6*l1 + 0.2*l2 + 0.2*ssim_val
        return loss



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
        
        print("[DEBUG forward] rec_img:", rec_img.shape,
          "target:", target.shape,
          "gt:", gt.shape,
          "y:", y.shape)
        
        # Fix per ground truth mask (gt)
        # Alcuni dataset (es. MNIST riscalato per ViT) generano gt con shape "storta"
        # come (N,224,1,224) invece di (N,1,224,224). Questo causa mismatch con rec_img.
        if gt.ndim == 4 and gt.shape[1] == 224 and gt.shape[2] == 1 and gt.shape[3] == 224:
            gt = gt.permute(0, 2, 1, 3)  # (N,224,1,224) → (N,1,224,224)

        # Se rec_img è RGB (3 canali) e gt è 1 canale, duplica gt su 3 canali
        if gt.shape[1] == 1 and rec_img.shape[1] == 3:
            gt = gt.repeat(1, 3, 1, 1)
        
        max_diff = (self.f(target) - target) ** 2

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
    
    
class AEXAD_loss_ViT(nn.Module): #FORSE DA ELIMINARE - vecchia loss introdotta all'inizio per il ViT
    def __init__(self, lambda_p=None, lambda_s=None,
                 f=lambda x: torch.where(x >= 0.5, 0.0, 1.0),
                 cuda=True):
        """
        Variante di AEXAD_loss_norm con clamp e debug print
        pensata per encoder diversi (es. ViT) dove le instabilità
        numeriche sono più probabili.
        """
        super().__init__()
        self.lambda_p = lambda_p
        self.lambda_s = lambda_s
        self.f = f
        self.use_cuda = cuda

        if lambda_p is not None:
            self.lambda_p = torch.tensor(lambda_p, dtype=torch.float32)
            if cuda:
                self.lambda_p = self.lambda_p.cuda()
        else:
            self.lambda_p = None

    def forward(self, rec_img, target, gt, y):
        print("[DEBUG forward] rec_img:", rec_img.shape,
              "target:", target.shape,
              "gt:", gt.shape,
              "y:", y.shape)

        # Fix shape ground truth
        if gt.ndim == 4 and gt.shape[1] == 224 and gt.shape[2] == 1 and gt.shape[3] == 224:
            print("[DEBUG] permuting GT shape (N,224,1,224) -> (N,1,224,224)")
            gt = gt.permute(0, 2, 1, 3)

        if gt.shape[1] == 1 and rec_img.shape[1] == 3:
            print("[DEBUG] duplicating GT channels to match RGB")
            gt = gt.repeat(1, 3, 1, 1)

        # --- max_diff ---
        max_diff = (self.f(target) - target) ** 2
        if torch.any(max_diff == 0):
            print("[DEBUG] max_diff contains zeros -> applying clamp")
        max_diff = torch.clamp(max_diff, min=1e-6)

        # --- reconstruction losses ---
        rec_n = (rec_img - target) ** 2 / max_diff
        rec_o = (self.f(target) - rec_img) ** 2 / max_diff

        # --- lambda_p ---
        if self.lambda_p is None:
            denom = torch.sum(gt, dim=(1, 2, 3))
            if torch.any(denom == 0):
                print("[DEBUG] denom contains zero (batch with no anomalies) -> applying clamp")
            denom = torch.clamp(denom, min=1.0)

            lambda_p = torch.reshape(
                np.prod(gt.shape[1:]) / denom, (-1, 1)
            )
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

        # --- final loss ---
        loss_vec = (1 - gt) * rec_n + lambda_p * gt * rec_o

        loss = torch.sum(loss_vec, dim=(1, 2, 3))
        loss_n = torch.sum((1 - gt) * rec_n) / torch.sum((1 - gt))
        loss_a = torch.sum(lambda_p * gt * rec_o) / torch.sum(lambda_p * gt)

        weighted_loss = torch.mean(loss)

        if torch.isnan(weighted_loss).any() or torch.isinf(weighted_loss).any():
            print("[DEBUG] weighted_loss is NaN/Inf!")

        return weighted_loss, loss_n, loss_a


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