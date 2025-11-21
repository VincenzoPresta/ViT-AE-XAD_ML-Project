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
        
        self.up1 = nn.Upsample(scale_factor=2)
        self.up2 = nn.Upsample(scale_factor=2)
        self.up3 = nn.Upsample(scale_factor=2)
        self.tan3 = nn.Tanh()

        # =============================
        #          ENCODER
        # =============================

        self.encoder = ViT_Encoder()

        # Esponiamo gli attributi necessari al Trainer
        self.conv_proj   = self.encoder.conv_proj
        self.class_token = self.encoder.class_token
        self.encoder_vit = self.encoder.encoder_vit
        self.to_64       = self.encoder.to_64
        self.up_to_28    = self.encoder.up_to_28

        # =============================
        #          DECODER
        # =============================

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.SELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SELU()
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.SELU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SELU()
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.SELU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.SELU()
        )

        self.decoder_final = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1),
            nn.SELU(),
            nn.Conv2d(8, dim[0], 3, padding=1),
            nn.Sigmoid()
        )

        # Decoder finale
        self.decoder = nn.Sequential(
            self.dec1,
            self.dec2,
            self.dec3,
            self.decoder_final
        )


    def forward(self, x):

        # ===== 1) ENCODER =====
        encoded = self.encoder(x)   # (B,64,28,28)

        # ===== 2) DECODER =====
        x = self.dec1(encoded)      # (B,32,56,56)
        x = self.dec2(x)            # (B,16,112,112)
        x = self.dec3(x)            # (B,8,224,224)

        # ===== 3) MASK MODULATION (identica all’originale) =====
        # upsample encoded 3 volte (senza conv) per la mask
        up = self.up1(encoded)
        up = self.up2(up)
        up = self.up3(up)

        mask = torch.sum(self.tan3(up)**2, axis=1).unsqueeze(1)
        x = x + x * mask

        # ===== 4) Final conv layer =====
        out = self.decoder_final(x)

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
 