import torch
import torch.nn as nn
import numpy as np
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch.nn.functional as F

class ViT_CNN_Attn(nn.Module):
    """
    AE-XAD con:
    - encoder Vision Transformer (classe ViT_Encoder)
    - decoder CNN identico a ResNet_CNN_Attn
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.encoder = ViT_Encoder()
        self.decoder = AEXAD_Decoder()

        # Esponiamo gli attributi necessari al Trainer
        self.conv_proj   = self.encoder.conv_proj
        self.class_token = self.encoder.class_token
        self.encoder_vit = self.encoder.encoder_vit
        self.to_64       = self.encoder.to_64
        self.up_to_28    = self.encoder.up_to_28

        
    def forward(self, x):

        encoded = self.encoder(x)   # (B,64,28,28)
        
        out = self.decoder(encoded)

        return out
    
    
class AEXAD_Decoder(nn.Module):
    def __init__(self, out_channels=3):
        super().__init__()

        # -------------------------------------------
        # BRANCH 1 (NON–TRAINABLE)
        # -------------------------------------------
        self.up1 = nn.Upsample(scale_factor=2)
        self.up2 = nn.Upsample(scale_factor=2)
        self.up3 = nn.Upsample(scale_factor=2)
        self.tanh = nn.Tanh()

        # NOTA: nessun parametro in Branch 1 viene appreso
        for p in self.up1.parameters(): p.requires_grad = False
        for p in self.up2.parameters(): p.requires_grad = False
        for p in self.up3.parameters(): p.requires_grad = False

        # -------------------------------------------
        # BRANCH 2 (TRAINABLE)
        # -------------------------------------------
        # 28×28×64 → 56×56×32
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.SELU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.SELU(),
        )

        # 56×56×32 → 112×112×16
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.SELU(),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.SELU(),
        )

        # 112×112×16 → 224×224×8
        self.dec3 = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.SELU(),
            nn.ConvTranspose2d(8, 8, kernel_size=4, stride=2, padding=1),
            nn.SELU(),
        )

        # -------------------------------------------
        # FINAL CONV (8 → 3)
        # -------------------------------------------
        self.final = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1),
            nn.SELU(),
            nn.Conv2d(8, out_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, encoded):
        B, C, H, W = encoded.shape  # (B,64,28,28)

        # ===========================
        # BRANCH 1 - NON TRAINABILE
        # ===========================
        b1 = self.up1(encoded)
        b1 = self.up2(b1)
        b1 = self.up3(b1)              # (B,64,224,224)
        b1 = self.tanh(b1)

        # somma gruppi di 8 canali: 64 → 8
        # ogni canale i = somma(encoded[ i*8 : (i+1)*8 ])
        b1 = b1.view(B, 8, 8, 224, 224).sum(dim=2)   # (B,8,224,224)

        # ===========================
        # BRANCH 2 - TRAINABILE
        # ===========================
        b2 = self.dec1(encoded)        # (B,32,56,56)
        b2 = self.dec2(b2)             # (B,16,112,112)
        b2 = self.dec3(b2)             # (B,8,224,224)

        # ===========================
        # MASK MODULATION
        # ===========================
        fused = b2 + b2 * b1           # (B,8,224,224)

        # ===========================
        # OUTPUT FINALE
        # ===========================
        out = self.final(fused)        # (B,3,224,224)
        return out



class ViT_Encoder(nn.Module):
    """
    Encoder ViT che sostituisce ResNet nel modello AE-XAD,
    producendo un tensore (B, 64, 28, 28) compatibile con il decoder originale.
    """
    def __init__(self):
        super().__init__()

        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        self.hidden_dim = vit.hidden_dim     # 768
        self.patch_size = vit.patch_size     # 16
        self.image_size = vit.image_size     # 224

        self.conv_proj   = vit.conv_proj
        self.class_token = vit.class_token
        self.encoder_vit = vit.encoder

        self.to_64 = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.LayerNorm(64)
        )
        
        self.up_to_28 = nn.Upsample(scale_factor=2)
        
        # ============================
        # FREEZING DELLA PATCH EMBEDDING
        # ============================
        for param in vit.conv_proj.parameters():
            param.requires_grad = False
        
        # ============================
        # FREEZING DEI PRIMI 6 BLOCCHI ViT
        # ============================
        for block in vit.encoder.layers[:6]:
            for param in block.parameters():
                param.requires_grad = False

    def _process_input(self, x):
        B = x.size(0)
        x = self.conv_proj(x)                # (B,768,14,14)
        x = x.reshape(B, self.hidden_dim, -1) # (B,768,196)
        x = x.permute(0, 2, 1)                # (B,196,768)
        return x

    def forward(self, x):
        B = x.size(0)

        tokens = self._process_input(x)

        cls = self.class_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        encoded = self.encoder_vit(tokens)[:, 1:]    # (B,196,768)

        encoded = self.to_64(encoded)                # (B,196,64)

        encoded = encoded.reshape(B, 14, 14, 64).permute(0, 3, 1, 2)
        encoded = self.up_to_28(encoded)             # (B,64,28,28)

        return encoded

#-------------------------------------------------------------------
 