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

        # ======== FIX GT ========
        # Se GT ha shape (B,224,1,224) → sistemala
        if gt.ndim == 4 and gt.shape[1] == H and gt.shape[2] == 1:
            gt = gt.permute(0, 2, 1, 3)

        # Se GT ha più canali → collapsalo a 1 canale
        if gt.shape[1] != 1:
            gt = torch.sum(gt, dim=1, keepdim=True)
            gt = (gt > 0).float()

        # Replica GT a 3 canali
        if C == 3:
            gt = gt.repeat(1, 3, 1, 1)

        gt = gt.to(device)

        # ======== DENOMINATORE ========
        F_x = 2.0
        max_diff = (F_x - target)**2
        max_diff = torch.clamp(max_diff, min=1e-6)

        # ======== RICOSTRUZIONI ========
        rec_n = (rec_img - target)**2 / max_diff
        rec_o = (F_x - rec_img)**2 / max_diff

        # ======== LAMBDA_p ========
        anomaly_pixels = torch.sum(gt, dim=(1,2,3))
        anomaly_pixels = torch.clamp(anomaly_pixels, min=1.0)
        

        lambda_p = (D / anomaly_pixels).to(device)      # shape: (B,)
        lambda_p = lambda_p.view(B, 1, 1, 1)            # shape: (B,1,1,1)
        lambda_p = lambda_p.repeat(1, C, H, W)          # shape: (B,3,224,224)

        # ======== LOSS ========
        
        loss_vec = (1 - gt) * rec_n + lambda_p * gt * rec_o

        loss = torch.mean(torch.sum(loss_vec, dim=(1,2,3)))
        return loss
