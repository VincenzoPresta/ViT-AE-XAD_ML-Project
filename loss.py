import torch
import torch.nn as nn

class AEXAD_Loss(nn.Module):
    """
    Implementazione ufficiale Eq.(1) AE-XAD Arrays.
    Versione pulita, stateless, con debug opzionale.
    """
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug

    def forward(self, rec_img, target, gt, y=None):

        device = rec_img.device
        B, C, H, W = target.shape
        D = C * H * W

        # ============================================================
        # GT FIX
        # ============================================================

        gt = gt.float()

        # (B,H,W,1) → (B,1,H,W)
        if gt.ndim == 4 and gt.shape[-1] == 1:
            gt = gt.permute(0,3,1,2).contiguous()

        # binarizzazione 0/255 → 0/1
        gt = (gt >= 128).float().to(device)

        # replica GT su 3 canali se necessario
        if C == 3:
            gt = gt.repeat(1, 3, 1, 1).contiguous()

        # ============================================================
        # AE-XAD LOSS TERMS
        # ============================================================

        # F(x) del paper
        F_x = 2.0

        # denominatore
        denom = (F_x - target)**2 + 1e-6

        # termini ricostruzione
        rec_normal = ((rec_img - target)**2) / denom
        rec_anom   = ((F_x - rec_img)**2)   / denom

        # λ_p = D / (# pixel anomali)
        anomaly_pixels = torch.sum(gt[:,0], dim=(1,2))   # usa solo 1 canale
        anomaly_pixels = torch.clamp(anomaly_pixels, min=1.0)

        lambda_p = (D / anomaly_pixels).view(B,1,1,1).to(device)
        lambda_p = lambda_p.repeat(1, C, H, W)

        # loss finale
        loss_vec = (1 - gt) * rec_normal + lambda_p * gt * rec_anom
        loss = torch.mean(torch.sum(loss_vec, dim=(1,2,3)))

        # ============================================================
        # DEBUG (STATeless)
        # ============================================================
        if self.debug:
            print("\n[DEBUG AEXAD LOSS]")
            print("gt unique:", torch.unique(gt))
            print("rec_normal mean:", rec_normal.mean().item())
            print("rec_anom mean  :", rec_anom.mean().item())
            print("lambda_p sample:", lambda_p.view(B, -1)[0, :5])
            print("loss:", loss.item())
            print("--------------------------------------------------")

        return loss
