import torch
import torch.nn as nn


class AEXAD_Loss(nn.Module):
    """
    Implementazione IDENTICA alla Eq.(1) di AE-XAD Arrays.
    Compatibile con RGB, GT monocanale, F(x)=2, lambda_y corretta.
    """

    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug

    def forward(self, rec_img, target, gt, y=None):
        """
        rec_img : output AE (B,3,H,W)
        target  : input originale (B,3,H,W)
        gt      : heatmap GT (B,1,H,W)  — deve essere 0/1
        """

        device = rec_img.device
        B, C, H, W = target.shape
        D = H * W  # numero di pixel (come nel paper)

        # ============================================================
        #           NORMALIZZAZIONE GT  (paper: yi ∈ {0,1})
        # ============================================================
        gt = gt.float().to(device)

        # (B,H,W,1) → (B,1,H,W)
        if gt.ndim == 4 and gt.shape[-1] == 1:
            gt = gt.permute(0, 3, 1, 2).contiguous()

        # binarizzazione 0–255 → 0–1
        gt = (gt >= 0.5).float()

        # GT rimane 1-channel (come nel paper)
        # NIENTE repeat sui 3 canali

        # ============================================================
        #                F(x) = v = 2  (paper)
        # ============================================================
        F_x = 2.0

        # ============================================================
        #   COSTRUZIONE DENOMINATORE PER-PIXEL (SOMMA SUI CANALI)
        #   denom_j = Σ_c  (F_x − x_{j,c})^2
        # ============================================================

        denom = torch.sum((F_x - target) ** 2, dim=1, keepdim=True) + 1e-6
        # shape: (B,1,H,W)

        # ============================================================
        #     TERMINE NORMAL (1 - y_j) * (x - x̃)^2 / denom_j
        # ============================================================
        num_normal = torch.sum((rec_img - target) ** 2, dim=1, keepdim=True)
        rec_normal = num_normal / denom
        # shape: (B,1,H,W)

        # ============================================================
        #   TERMINE ANOMALO y_j * (F(x) − x̃)^2 / denom_j
        # ============================================================

        num_anom = torch.sum((F_x - rec_img) ** 2, dim=1, keepdim=True)
        rec_anom = num_anom / denom
        # shape: (B,1,H,W)

        # ============================================================
        #            λ_y = D / (# pixel anomali)
        # ============================================================

        anomaly_pixels = torch.clamp(gt.sum(dim=(1, 2, 3)), min=1.0)
        lambda_y = (D / anomaly_pixels).view(B, 1, 1, 1)
        
        # ============================================================
        #                    LOSS FINALE
        # ============================================================
        loss_pixel = (1 - gt) * rec_normal + lambda_y * gt * rec_anom
        loss = loss_pixel.sum(dim=(1, 2, 3)).mean()
        print("Loss baseline AE-XAD attiva")

        # ============================================================
        #                         DEBUG
        # ============================================================
        if self.debug:
            print("\n[DEBUG AEXAD LOSS]")
            print("gt unique:", torch.unique(gt))
            print("rec_normal mean:", rec_normal.mean().item())
            print("rec_anom mean  :", rec_anom.mean().item())
            print("lambda_y:", lambda_y.view(-1)[:5])
            print("loss:", loss.item())
            print("--------------------------------------------------")

        return loss
