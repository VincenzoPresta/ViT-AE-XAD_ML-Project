import torch
import torch.nn as nn
import numpy as np
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch.nn.functional as F

class ViT_CNN_Attn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = ViT_Encoder()
        self.decoder = AEXAD_Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        out = self.decoder(encoded)
        return out
    
    
class AEXAD_Decoder(nn.Module):
    def __init__(self, out_channels=3):
        super().__init__()

        # ============================================================
        # BRANCH 1 — NON TRAINABLE (paper)
        # ============================================================

        # Un unico upsample 28 → 224, NON tre consecutivi
        self.up = nn.Upsample(size=(224,224), mode="nearest")
        for p in self.up.parameters():
            p.requires_grad = False

        self.tanh = nn.Tanh()

        # ============================================================
        # BRANCH 2 — TRAINABLE (paper)
        # ============================================================

        # 28×28×64 → 56×56×32
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.SELU(),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.SELU(),
        )

        # 56×56×32 → 112×112×16
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.SELU(),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.SELU(),
        )

        # 112×112×16 → 224×224×8
        self.dec3 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.SELU(),
            nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2),
            nn.SELU(),
        )

        # FINAL 8 → 3
        self.final = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.SELU(),
            nn.Conv2d(8, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape  # (B,64,28,28)

        # ===========================
        # BRANCH 1 (paper)
        # ===========================
        b1 = self.up(x)               # → (B,64,224,224)
        b1 = self.tanh(b1)
        b1 = b1.view(B, 8, 8, 224, 224).sum(dim=2)  # 64→8

        # ===========================
        # BRANCH 2 (paper)
        # ===========================
        b2 = self.dec1(x)
        b2 = self.dec2(b2)
        b2 = self.dec3(b2)

        # ===========================
        # MODULATION
        # ===========================
        fused = b2 + b1 * b2

        # ===========================
        # FINAL
        # ===========================
        out = self.final(fused)
        return out



class ViT_Encoder(nn.Module):
    """
    Encoder ViT per AE-XAD con stem convoluzionale:
    - stem conv prima del ViT (cattura bordi/texture fini)
    - usa patch embedding e blocchi ViT
    - ricostruisce struttura spaziale via CNN
    - genera feature map 28×28×64 compatibili col decoder AE-XAD
    """

    def __init__(self, freeze_vit: bool = True):
        super().__init__()

        # ===== STEM CONVOLUZIONALE PRIMA DEL VIT =====
        # mantiene la risoluzione 224×224 e torna a 3 canali
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

        # ===== VIT ORIGINALE =====
        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        self.hidden_dim   = vit.hidden_dim  # 768
        self.conv_proj    = vit.conv_proj   # patch embedding
        self.encoder_vit  = vit.encoder
        self.class_token  = vit.class_token

        # ===== FREEZE SOLO IL VIT (OPZIONALE) =====
        if freeze_vit:
            for p in self.conv_proj.parameters():
                p.requires_grad = False
            for p in self.encoder_vit.parameters():
                p.requires_grad = False
            if isinstance(self.class_token, nn.Parameter):
                self.class_token.requires_grad = False

        # ======= RICOSTRUZIONE SPAZIALE =========
        self.to_spatial = nn.Sequential(
            nn.Conv2d(self.hidden_dim, 256, kernel_size=1),
            nn.SELU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.SELU()
        )

        # ======= UPSAMPLE A 28×28 =========
        self.to_28 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.SELU()
        )

    def _patchify(self, x):
        # x: (B,3,224,224)
        B = x.size(0)
        x = self.conv_proj(x)                  # (B,768,14,14)
        x = x.reshape(B, self.hidden_dim, -1)  # (B,768,196)
        x = x.permute(0, 2, 1)                 # (B,196,768)
        return x

    def forward(self, x):
        B = x.size(0)

        # ===== STEM + VIT =====
        x = self.stem(x)        # (B,3,224,224) -> (B,3,224,224) arricchita di dettagli

        tokens = self._patchify(x)  # (B,196,768)

        cls = self.class_token.expand(B, -1, -1)  # (B,1,768)
        tokens = torch.cat([cls, tokens], dim=1)  # (B,197,768)

        encoded = self.encoder_vit(tokens)[:, 1:]  # (B,196,768), rimuovi CLS
        encoded = encoded.view(B, 14, 14, self.hidden_dim)
        encoded = encoded.permute(0, 3, 1, 2)      # (B,768,14,14)

        # ===== CNN RECONSTRUCTION =====
        spatial = self.to_spatial(encoded)         # (B,128,14,14)
        out = self.to_28(spatial)                  # (B,64,28,28)

        return out
