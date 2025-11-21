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

#-------------------------#

class AEXAD_Loss(nn.Module):
    """
    Implementazione ufficiale AE-XAD (Eq. 1 del paper), con:
    - F(x) = 2            (versione originale del paper)
    - λ_p = D / (#anomali) (per-pixel weight dal paper)
    - Normalizzazione per-pixel (F(x)-x)^2
    - Supporto automatico per GT in 1 canale o shape invertita
    """

    def __init__(self):
        super().__init__()

    def forward(self, rec_img, target, gt, y):
        """
        rec_img: output del decoder (B,3,H,W)
        target:  input image (B,3,H,W)
        gt:      ground-truth mask (B,1,H,W) oppure (B,3,H,W)
        y:       label per immagine (non usato dal paper ma tenuto per compatibilità)
        """

        B, C, H, W = target.shape
        D = H * W * C

        # ===== Fix GT shape (automatico) =====
        if gt.ndim == 4 and gt.shape[1] == H and gt.shape[2] == 1 and gt.shape[3] == W:
            gt = gt.permute(0, 2, 1, 3)

        # Se GT è a 1 canale, estendila a 3 canali
        if gt.shape[1] == 1 and C == 3:
            gt = gt.repeat(1, 3, 1, 1)

        # ===== Denominatore del paper =====
        F_x = 2.0
        max_diff = (F_x - target) ** 2
        max_diff = torch.clamp(max_diff, min=1e-6)

        # ===== Termini della loss =====
        rec_n = (rec_img - target) ** 2 / max_diff
        rec_o = (F_x - rec_img) ** 2 / max_diff

        # ===== λ_p = D / (#anomali) =====
        anomaly_pixels = torch.sum(gt, dim=(1, 2, 3))
        anomaly_pixels = torch.clamp(anomaly_pixels, min=1.0)

        lambda_p = (D / anomaly_pixels).view(B, 1, 1)
        lambda_p = lambda_p.repeat(1, C, H, W)

        # ===== Loss per pixel =====
        loss_vec = (1 - gt) * rec_n + lambda_p * gt * rec_o

        # ===== Loss finale (batch-mean) =====
        loss = torch.mean(torch.sum(loss_vec, dim=(1, 2, 3)))

        return loss