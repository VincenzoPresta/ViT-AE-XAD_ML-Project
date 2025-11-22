import torch
import torch.nn as nn
import torch.nn.functional as F

class AEXAD_Loss(nn.Module):
    """
    Implementazione ufficiale AE-XAD (Eq. 1).
    Compatibile con ViT.
    """
    def __init__(self):
        super().__init__()

    def forward(self, rec_img, target, gt, y):

        device = rec_img.device
        B, C, H, W = target.shape
        D = C * H * W

        # ======== GT FIX ========
        gt = gt.float()

        # (B, H, W, 1) → (B,1,H,W)
        if gt.ndim == 4 and gt.shape[-1] == 1:
            gt = gt.permute(0, 3, 1, 2)

        # binaria
        gt = (gt > 0).float().to(device)

        # replica GT a 3 canali
        if C == 3:
            gt = gt.repeat(1, 3, 1, 1)

        # ======= AE-XAD loss ufficiale ========
        F_x = 2.0
        denom = (F_x - target)**2 + 1e-6

        rec_normal = (rec_img - target)**2 / denom
        rec_anom   = (F_x - rec_img)**2 / denom

        anomaly_pixels = torch.sum(gt, dim=(1,2,3))
        anomaly_pixels = torch.clamp(anomaly_pixels, min=1.0)

        lambda_p = (D / anomaly_pixels).view(B,1,1,1).to(device)

        loss_vec = (1 - gt) * rec_normal + lambda_p * gt * rec_anom

        loss = torch.mean(torch.sum(loss_vec, dim=(1,2,3)))
        return loss

