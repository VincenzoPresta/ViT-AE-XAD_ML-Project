import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from torchvision.models import vgg11_bn, resnet50, vit_b_32, vit_b_16, inception_v3
import torch.nn.functional as F

class ViT_CNN_Attn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Carichiamo il ViT base (no pretraining ora)
        vit = vit_b_16(weights=None)

        self.image_size = vit.image_size    # 224
        self.hidden_dim = vit.hidden_dim    # 768
        self.patch_size = vit.patch_size    # 16

        self.conv_proj = vit.conv_proj      # patch embedding convoluzione
        self.class_token = vit.class_token
        self.encoder = vit.encoder          # Encoder Transformer

        # Il decoder MAE riceve token 768-dim
        self.decoder = MAEDecoder(
            patch_dim=self.hidden_dim,
            dec_dim=256
        )

    def _process_input(self, x):
        n, c, h, w = x.shape
        p = self.patch_size
        
        torch._assert(h == self.image_size, "Wrong image height!")
        torch._assert(w == self.image_size, "Wrong image width!")

        n_h = h // p
        n_w = w // p

        # Conv patch embedding → (N, 768, 14, 14)
        x = self.conv_proj(x)

        # Flatten → (N, 768, 196)
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # Transpose in formato ViT → (N, 196, 768)
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x):
        x = self._process_input(x)
        B, S, D = x.shape

        # Aggiungi class token
        cls = self.class_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Encoder ViT
        enc = self.encoder(x)

        # Rimuovi class token → (B, 196, 768)
        enc = enc[:, 1:]

        # Decodifica MAE → immagine ricostruita
        out = self.decoder(enc)
        return out
    
    
class MAEDecoder(nn.Module):
    def __init__(self, patch_dim=768, dec_dim=256):
        super().__init__()

        # Proiezione dei token ViT nel latent decoder
        self.proj = nn.Sequential(
            nn.Linear(patch_dim, 512),
            nn.GELU(),
            nn.Linear(512, dec_dim)
        )

        # Decoder piramidale stile MAE
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(dec_dim, 256, kernel_size=2, stride=2),  # 14 → 28
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),      # 28 → 56
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),       # 56 → 112
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),        # 112 → 224
            nn.GELU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, tokens):
        B, N, D = tokens.shape
        H = W = int(N ** 0.5)  # 196 → 14×14

        x = self.proj(tokens)                 # (B,196,256)
        x = x.permute(0,2,1).reshape(B,256,H,W)  # (B,256,14,14)
        x = self.deconv(x)                    # (B,3,224,224)
        return x

#--------------------------------------------------------------------------------#

class Shallow_Autoencoder(nn.Module):
    def __init__(self, dim, flat_dim, latent_dim):
        super(Shallow_Autoencoder, self).__init__()
        self.dim = dim

        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, flat_dim),
            nn.Sigmoid(),
        )


    def forward(self, x):
        x_f = x.flatten(start_dim=1)
        encoded = self.encoder(x_f)
        decoded = self.decoder(encoded)
        decoded = torch.reshape(decoded, x.shape)
        return decoded


class Deep_Autoencoder(nn.Module):
    def __init__(self, dim, flat_dim, intermediate_dim, latent_dim):
        super(Deep_Autoencoder, self).__init__()
        self.dim = dim

        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, flat_dim),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x_f = x.flatten(start_dim=1)
        encoded = self.encoder(x_f)
        decoded = self.decoder(encoded)
        decoded = torch.reshape(decoded, x.shape)
        return decoded


class Conv_Deep_Autoencoder(nn.Module):
    def __init__(self, dim):
        super(Conv_Deep_Autoencoder, self).__init__()
        self.dim = np.array(dim)
        print(dim)

        layers = []
        diffs = []

        layers.append(nn.Conv2d(dim[0], 16, (5, 5), stride=1, padding=2))
        mods = np.remainder(-self.dim[1:], 4)
        diff = np.array([mods[-2] // 2, mods[-2] - mods[-2] // 2, mods[-1] // 2, mods[-1] - mods[-1] // 2])
        if np.sum(diff) > 0:
            layers.append(nn.ZeroPad2d(diff))
        diffs.append(diff)
        layers.append(nn.MaxPool2d((4, 4), stride=4))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(16, 32, (5, 5), stride=1, padding=2))
        mods = np.remainder(-np.floor_divide(self.dim[1:], 4), 4)
        diff = np.array([mods[-2] // 2, mods[-2] - mods[-2] // 2, mods[-1] // 2, mods[-1] - mods[-1] // 2])
        if np.sum(diff) > 0:
            layers.append(nn.ZeroPad2d(diff))
        diffs.append(diff)
        layers.append(nn.MaxPool2d((4, 4), stride=4))
        layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)

        layers = []

        if np.sum(diffs[0]) > 0:
            layers.append(nn.ZeroPad2d(-diffs[0]))
        layers.append(nn.Upsample(scale_factor=4))
        layers.append(nn.Conv2d(32, 32, (5, 5), stride=1, padding=2))
        layers.append(nn.ReLU())
        if np.sum(diffs[1]) > 0:
            layers.append(nn.ZeroPad2d(-diffs[1]))
        layers.append(nn.Upsample(scale_factor=4))
        layers.append(nn.Conv2d(32, 16, (5, 5), stride=1, padding=2))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(16, dim[0], (5, 5), padding=2))
        layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



class Conv_Deep_Autoencoder_v2(nn.Module):
    def __init__(self, dim):
        super(Conv_Deep_Autoencoder_v2, self).__init__()
        self.dim = np.array(dim)
        print(dim)

        layers = []

        layers.append(nn.Conv2d(dim[0], 16, (5, 5), stride=1, padding=2))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(16, 16, (5, 5), stride=1, padding=2))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(16, 32, (6, 6), stride=4, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())

        layers.append(nn.Conv2d(32, 32, (5, 5), stride=1, padding=2))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(32, 32, (5, 5), stride=1, padding=2))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(32, 128, (6, 6), stride=4, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU())



        self.encoder = nn.Sequential(*layers)

        layers = []

        layers.append(nn.ConvTranspose2d(128, 32, (6, 6), stride=4, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(32, 32, (5, 5), stride=1, padding=2))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(32, 32, (5, 5), stride=1, padding=2))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())

        layers.append(nn.ConvTranspose2d(32, 16, (6, 6), stride=4, padding=1))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(16, 16, (5, 5), stride=1, padding=2))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(16, 16, (5, 5), stride=1, padding=2))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(16, dim[0], (3, 3), stride=1, padding=1))
        layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Conv_Deep_mask(nn.Module):
    def __init__(self, dim):
        super(Conv_Deep_mask, self).__init__()
        self.dim = dim

        layers = []

        layers.append(MultiScaleFeature(dim[0], 16, dilatation=1, activation='relu'))
        layers.append(MultiScaleFeature(16, 16, dilatation=2, activation='relu'))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.Conv2d(16, 32, (4, 4), stride=2, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())

        layers.append(MultiScaleFeature(32, 32, dilatation=1, activation='relu'))
        layers.append(MultiScaleFeature(32, 32, dilatation=2, activation='relu'))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.Conv2d(32, 64, (4, 4), stride=2, padding=1))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU())

        layers.append(MultiScaleFeature(64, 64, dilatation=1, activation='relu'))
        layers.append(MultiScaleFeature(64, 64, dilatation=2, activation='relu'))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.Conv2d(64, 128, (4, 4), stride=2, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU())

        layers.append(nn.Conv2d(128, 128, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(128, 64, (1, 1), stride=1, padding=0))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)

        self.up1 = nn.Upsample(scale_factor=2)
        self.tan = nn.Tanh()

        self.up2 = nn.Upsample(scale_factor=2)
        #self.tan2 = nn.Tanh()

        self.up3 = nn.Upsample(scale_factor=2)
        #self.tan3 = nn.Tanh()

        layers = []

        layers.append(nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1))
        layers.append(nn.SELU())
        #layers.append(nn.Conv2d(32, 32, (3, 3), dilation=2, stride=1, padding=2))
        #layers.append(nn.SELU())
        layers.append(MultiScaleFeature(32, 32))
        layers.append(MultiScaleFeature(32, 32))

        self.dec1 = nn.Sequential(*layers)

        layers = []

        layers.append(nn.ConvTranspose2d(32, 16, (4, 4), stride=2, padding=1))
        # layers.append(nn.BatchNorm2d(16))
        layers.append(nn.SELU())
        #layers.append(nn.Conv2d(16, 16, (3, 3), dilation=2, stride=1, padding=2))
        ##layers.append(nn.BatchNorm2d(32))
        #layers.append(nn.SELU())
        layers.append(MultiScaleFeature(16, 16))
        layers.append(MultiScaleFeature(16, 16))

        self.dec2 = nn.Sequential(*layers)

        layers = []

        layers.append(nn.ConvTranspose2d(16, 8, (4, 4), stride=2, padding=1))
        # layers.append(nn.BatchNorm2d(8))
        layers.append(nn.SELU())
        #layers.append(nn.Conv2d(8, 8, (3, 3), dilation=2, stride=1, padding=2))
        ##layers.append(nn.BatchNorm2d(16))
        #layers.append(nn.SELU())
        layers.append(MultiScaleFeature(8, 8))
        layers.append(MultiScaleFeature(8, 8))

        #layers.append(nn.ConvTranspose2d(32, 16,  (6, 6), stride=2, padding=2))
        #layers.append(nn.BatchNorm2d(16))
        #layers.append(nn.ReLU())
        #layers.append(nn.Conv2d(16, 16, (5, 5), stride=1, padding=2))
        #layers.append(nn.BatchNorm2d(16))
        #layers.append(nn.ReLU())
        #layers.append(nn.Conv2d(16, 16, (5, 5), stride=1, padding=2))
        #layers.append(nn.BatchNorm2d(16))
        #layers.append(nn.ReLU())

        #layers.append(nn.Conv2d(32, 16, (3, 3), stride=1, padding=1))
        #layers.append(nn.ReLU())
        #layers.append(nn.Conv2d(32, 32, (3, 3), stride=1, padding=1))
        #layers.append(nn.ReLU())
        #layers.append(nn.Conv2d(32, 16, (3, 3), stride=1, padding=1))

        self.dec3 = nn.Sequential(*layers)

        layers = []


        layers.append(nn.Conv2d(8, 8, (3, 3), stride=1, padding=1))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(8, 8, (3, 3), stride=1, padding=0))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(8, dim[0], (1, 1), stride=1, padding=1))
        layers.append(nn.Sigmoid())

        self.decoder_final = nn.Sequential(*layers)

    def forward(self, x):
        encoded = self.encoder(x)

        upsampled1 = self.up1(encoded)
        decoded1 = self.dec1(encoded)

        upsampled2 = self.up2(upsampled1)
        decoded2 = self.dec2(decoded1)

        upsampled3 = self.up3(upsampled2)
        decoded3 = self.dec3(decoded2) * torch.sum(self.tan(upsampled3), axis=1).unsqueeze(1)
        #decoded3 = self.dec3(decoded2) * upsampled3

        decoded = self.decoder_final(decoded3) # unsqueeze

        return decoded



class Conv_Autoencoder(nn.Module):
    def __init__(self, dim):
        super(Conv_Autoencoder, self).__init__()
        self.dim = dim
        print(dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(dim[0], 16, (3, 3), stride=1, padding=1), #In torch non c'è padding='same' come tensorflow per strides > 2
            nn.MaxPool2d((2,2), stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 8, (3, 3), stride=1, padding=1),
            nn.MaxPool2d((2, 2), stride=2),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8, 8, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            #nn.Conv2d(8, 8, (3, 3), padding=0),
            #nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8, 16, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, dim[0], (3, 3), padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class PCA_Autoencoder(nn.Module):
    def __init__(self, dim, flat_dim, latent_dim):
        super(PCA_Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.dim = dim

        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, latent_dim, bias=False),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, flat_dim, bias=False),
        )

    def forward(self, x):
        x_f = x.flatten(start_dim=1)
        encoded = self.encoder(x_f)
        decoded = self.decoder(encoded)
        decoded = torch.reshape(decoded, x.shape)
        return decoded


class Conv_Deep_Autoencoder_v2_Attn(nn.Module):
    def __init__(self, dim):
        super(Conv_Deep_Autoencoder_v2_Attn, self).__init__()
        self.dim = np.array(dim)
        print(dim)

        layers = []

        layers.append(nn.Conv2d(dim[0], 16, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.ReLU())
        layers.append(CBAM(16))
        layers.append(nn.Conv2d(16, 16, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(16, 32, (4, 4), stride=2, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())

        layers.append(nn.Conv2d(32, 32, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        layers.append(CBAM(32))
        layers.append(nn.Conv2d(32, 32, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(32, 64, (4, 4), stride=2, padding=1))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU())

        layers.append(nn.Conv2d(64, 64, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU())
        layers.append(CBAM(64))
        layers.append(nn.Conv2d(64, 64, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(64, 128, (4, 4), stride=2, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU())

        layers.append(nn.Conv2d(128, 128, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU())
        layers.append(CBAM(128))
        layers.append(nn.Conv2d(128, 128, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(128, 256, (4, 4), stride=2, padding=1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU())

        layers.append(CBAM(256))

        self.encoder = nn.Sequential(*layers)

        layers = []

        layers.append(nn.ConvTranspose2d(256, 128, (4, 4), stride=2, padding=1))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(128, 128, (3, 3), stride=1, padding=1))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(128, 128, (3, 3), stride=1, padding=1))
        layers.append(nn.SELU())

        layers.append(nn.ConvTranspose2d(128, 64, (4, 4), stride=2, padding=1))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(64, 64, (3, 3), stride=1, padding=1))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(64, 64, (3, 3), stride=1, padding=1))
        layers.append(nn.SELU())

        layers.append(nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(32, 32, (3, 3), stride=1, padding=1))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(32, 32, (3, 3), stride=1, padding=1))
        layers.append(nn.SELU())

        layers.append(nn.ConvTranspose2d(32, 16, (4, 4), stride=2, padding=1))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(16, 16, (3, 3), stride=1, padding=1))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(16, dim[0], (3, 3), stride=1, padding=1))
        layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class VGG_CNN(nn.Module):
    def __init__(self, dim):
        super(VGG_CNN, self).__init__()
        self.dim = dim
        self.encoder = vgg11_bn(pretrained=True).features[:16]
        for param in self.encoder.parameters():
            param.requires_grad = True

        layers = []

        layers.append(nn.ConvTranspose2d(512, 128, (6, 6), stride=4, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(128, 128, (5, 5), stride=1, padding=2))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(128, 128, (5, 5), stride=1, padding=2))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU())

        layers.append(nn.ConvTranspose2d(128, 32, (6, 6), stride=4, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(32, 32, (5, 5), stride=1, padding=2))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(32, 32, (5, 5), stride=1, padding=2))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())

        #layers.append(nn.ConvTranspose2d(32, 16,  (6, 6), stride=2, padding=2))
        #layers.append(nn.BatchNorm2d(16))
        #layers.append(nn.ReLU())
        #layers.append(nn.Conv2d(16, 16, (5, 5), stride=1, padding=2))
        #layers.append(nn.BatchNorm2d(16))
        #layers.append(nn.ReLU())
        #layers.append(nn.Conv2d(16, 16, (5, 5), stride=1, padding=2))
        #layers.append(nn.BatchNorm2d(16))
        #layers.append(nn.ReLU())

        #layers.append(nn.Conv2d(32, 16, (3, 3), stride=1, padding=1))
        #layers.append(nn.ReLU())
        #layers.append(nn.Conv2d(32, 32, (3, 3), stride=1, padding=1))
        #layers.append(nn.ReLU())
        #layers.append(nn.Conv2d(32, 16, (3, 3), stride=1, padding=1))

        layers.append(nn.Conv2d(32, 16, (3, 3), stride=1, padding=1))
        #layers.append(nn.BatchNorm2d(16))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(16, 16, (3, 3), stride=1, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(16, dim[0], (3, 3), stride=1, padding=1))
        layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class VGG_CNN_new(nn.Module):
    def __init__(self, dim):
        super(VGG_CNN_new, self).__init__()
        self.dim = dim
        self.encoder_pre = vgg11_bn(pretrained=True).features[:21]
        for param in self.encoder_pre.parameters():
            param.requires_grad = False

        layers = []

        layers.append(nn.Conv2d(512, 256, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(256, 128, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(128, 64, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)

        layers = []

        layers.append(nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(32, 32, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())

        layers.append(nn.ConvTranspose2d(32, 16, (4, 4), stride=2, padding=1))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(16, 16, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.ReLU())

        layers.append(nn.ConvTranspose2d(16, 8, (4, 4), stride=2, padding=1))
        layers.append(nn.BatchNorm2d(8))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(8, 8, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(8))
        layers.append(nn.ReLU())

        #layers.append(nn.ConvTranspose2d(32, 16,  (6, 6), stride=2, padding=2))
        #layers.append(nn.BatchNorm2d(16))
        #layers.append(nn.ReLU())
        #layers.append(nn.Conv2d(16, 16, (5, 5), stride=1, padding=2))
        #layers.append(nn.BatchNorm2d(16))
        #layers.append(nn.ReLU())
        #layers.append(nn.Conv2d(16, 16, (5, 5), stride=1, padding=2))
        #layers.append(nn.BatchNorm2d(16))
        #layers.append(nn.ReLU())

        #layers.append(nn.Conv2d(32, 16, (3, 3), stride=1, padding=1))
        #layers.append(nn.ReLU())
        #layers.append(nn.Conv2d(32, 32, (3, 3), stride=1, padding=1))
        #layers.append(nn.ReLU())
        #layers.append(nn.Conv2d(32, 16, (3, 3), stride=1, padding=1))

        layers.append(nn.Conv2d(8, 8, (3, 3), stride=1, padding=1))
        #layers.append(nn.BatchNorm2d(16))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(8, dim[0], (3, 3), stride=1, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(dim[0], dim[0], (3, 3), stride=1, padding=1))
        layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        fe = self.encoder_pre(x)
        encoded = self.encoder(fe)
        decoded = self.decoder(encoded)
        return decoded


class ResNet_CNN_Attn(nn.Module):
    def __init__(self, dim):
        super(ResNet_CNN_Attn, self).__init__()
        self.dim = dim
        encoder_pre = resnet50(pretrained=True)
        for param in encoder_pre.parameters():
            param.requires_grad = True

        self.conv1 = encoder_pre.conv1
        self.bn1 = encoder_pre.bn1
        self.relu1 = encoder_pre.relu
        self.maxpool1 = encoder_pre.maxpool
        self.layer1 = encoder_pre.layer1
        self.layer2 = encoder_pre.layer2

        layers = []

        layers.append(nn.Conv2d(512, 256, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(256, 128, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(128, 64, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(64))
        #layers.append(nn.ReLU())

        #self.att = CBAM(64)

        self.encoder = nn.Sequential(*layers)

        self.up1 = nn.Upsample(scale_factor=2)

        self.up2 = nn.Upsample(scale_factor=2)

        self.up3 = nn.Upsample(scale_factor=2)
        self.tan3 = nn.Tanh()

        layers = []

        layers.append(nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(32, 32, (3, 3), stride=1, padding=1))
        layers.append(nn.SELU())

        self.dec1 = nn.Sequential(*layers)

        layers = []

        layers.append(nn.ConvTranspose2d(32, 16, (4, 4), stride=2, padding=1))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(16, 16, (3, 3), stride=1, padding=1))
        layers.append(nn.SELU())

        self.dec2 = nn.Sequential(*layers)

        layers = []

        layers.append(nn.ConvTranspose2d(16, 8, (4, 4), stride=2, padding=1))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(8, 8, (3, 3), stride=1, padding=1))
        layers.append(nn.SELU())

        self.dec3 = nn.Sequential(*layers)

        layers = []

        layers.append(nn.Conv2d(8, 8, (3, 3), stride=1, padding=1))
        # layers.append(nn.BatchNorm2d(16))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(8, dim[0], (3, 3), stride=1, padding=1))
        #layers.append(nn.Sigmoid())
        layers.append(nn.ReLU())

        self.decoder_final = nn.Sequential(*layers)

    def forward(self, x):
        fe = self.conv1(x)
        fe = self.bn1(fe)
        fe = self.relu1(fe)
        fe = self.maxpool1(fe)
        fe1 = self.layer1(fe)
        fe2 = self.layer2(fe1)

        encoded = self.encoder(fe2)

        upsampled1 = self.up1(encoded)
        decoded1 = self.dec1(encoded)

        upsampled2 = self.up2(upsampled1)
        decoded2 = self.dec2(decoded1)

        upsampled3 = self.up3(upsampled2)
        decoded3 = self.dec3(decoded2)
        decoded3 = decoded3 + decoded3 * torch.sum(self.tan3(upsampled3)**2, axis=1).unsqueeze(1)

        decoded = self.decoder_final(decoded3)  # unsqueeze

        return decoded



class VGG_CNN_mask(nn.Module):
    def __init__(self, dim):
        super(VGG_CNN_mask, self).__init__()
        self.dim = dim
        self.encoder_pre = vgg11_bn(pretrained=True).features[:21]
        for param in self.encoder_pre.parameters():
            param.requires_grad = True

        layers = []

        layers.append(nn.Conv2d(512, 256, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(256, 128, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(128, 64, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(64))
        #layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)

        self.up1 = nn.Upsample(scale_factor=2)

        self.up2 = nn.Upsample(scale_factor=2)

        self.up3 = nn.Upsample(scale_factor=2)
        self.tan3 = nn.Tanh()

        layers = []

        layers.append(nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(32, 32, (3, 3), stride=1, padding=1))
        layers.append(nn.SELU())

        self.dec1 = nn.Sequential(*layers)

        layers = []

        layers.append(nn.ConvTranspose2d(32, 16, (4, 4), stride=2, padding=1))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(16, 16, (3, 3), stride=1, padding=1))
        layers.append(nn.SELU())

        self.dec2 = nn.Sequential(*layers)

        layers = []

        layers.append(nn.ConvTranspose2d(16, 8, (4, 4), stride=2, padding=1))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(8, 8, (3, 3), stride=1, padding=1))
        layers.append(nn.SELU())

        self.dec3 = nn.Sequential(*layers)

        layers = []

        layers.append(nn.Conv2d(8, 8, (3, 3), stride=1, padding=1))
        #layers.append(nn.BatchNorm2d(16))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(8, dim[0], (3, 3), stride=1, padding=1))
        layers.append(nn.Sigmoid())

        self.decoder_final = nn.Sequential(*layers)

    def forward(self, x):
        fe = self.encoder_pre(x)
        encoded = self.encoder(fe)

        upsampled1 = self.up1(encoded)
        decoded1 = self.dec1(encoded)

        upsampled2 = self.up2(upsampled1)
        decoded2 = self.dec2(decoded1)

        upsampled3 = self.up3(upsampled2)
        decoded3 = self.dec3(decoded2) * torch.sum(self.tan3(upsampled3), axis=1).unsqueeze(1)

        decoded = self.decoder_final(decoded3) # unsqueeze

        return decoded


class MultiScaleFeature(nn.Module):
    def __init__(self, channel_in, channel_out, dilatation=1, activation='selu'):
        super(MultiScaleFeature, self).__init__()

        self.conv1 = nn.Conv2d(channel_out, channel_out, kernel_size=(1, 1), stride=1, padding=0)
        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size=(3, 3), stride=1, padding=1*dilatation, dilation=dilatation)
        self.conv5 = nn.Conv2d(channel_in, channel_out, kernel_size=(5, 5), stride=1, padding=2*dilatation, dilation=dilatation)
        self.conv7 = nn.Conv2d(channel_in, channel_out, kernel_size=(7, 7), stride=1, padding=3*dilatation, dilation=dilatation)

        self.act = nn.ReLU()

    def forward(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)

        xf = x3 + x5 + x7
        #xf = self.bn1(xf)
        #xf = self.act1(xf)
        xf = self.conv1(xf)
        xf = self.act(xf)

        return xf



class ResNet_CNN_mask(nn.Module):
    def __init__(self, dim):
        super(ResNet_CNN_mask, self).__init__()
        self.dim = dim
        encoder_pre = resnet50(pretrained=True)
        for param in encoder_pre.parameters():
            param.requires_grad = True

        self.conv1 = encoder_pre.conv1
        self.bn1 = encoder_pre.bn1
        self.relu1 = encoder_pre.relu
        self.maxpool1 = encoder_pre.maxpool
        self.layer1 = encoder_pre.layer1
        self.layer2 = encoder_pre.layer2

        layers = []

        layers.append(nn.Conv2d(512, 256, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(256, 128, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(128, 64, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(64))
        #layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)

        self.up1 = nn.Upsample(scale_factor=2)
        self.tan = nn.Tanh()

        self.up2 = nn.Upsample(scale_factor=2)
        #self.tan2 = nn.Tanh()

        self.up3 = nn.Upsample(scale_factor=2)
        #self.tan3 = nn.Tanh()

        layers = []

        layers.append(nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1))
        layers.append(nn.SELU())
        #layers.append(nn.Conv2d(32, 32, (3, 3), dilation=2, stride=1, padding=2))
        #layers.append(nn.SELU())
        layers.append(MultiScaleFeature(32, 32, dilatation=2))
        layers.append(MultiScaleFeature(32, 32, dilatation=1))

        self.dec1 = nn.Sequential(*layers)

        layers = []

        layers.append(nn.ConvTranspose2d(32, 16, (4, 4), stride=2, padding=1))
        # layers.append(nn.BatchNorm2d(16))
        layers.append(nn.SELU())
        #layers.append(nn.Conv2d(16, 16, (3, 3), dilation=2, stride=1, padding=2))
        ##layers.append(nn.BatchNorm2d(32))
        #layers.append(nn.SELU())
        layers.append(MultiScaleFeature(16, 16, dilatation=2))
        layers.append(MultiScaleFeature(16, 16, dilatation=1))

        self.dec2 = nn.Sequential(*layers)

        layers = []

        layers.append(nn.ConvTranspose2d(16, 8, (4, 4), stride=2, padding=1))
        # layers.append(nn.BatchNorm2d(8))
        layers.append(nn.SELU())
        #layers.append(nn.Conv2d(8, 8, (3, 3), dilation=2, stride=1, padding=2))
        ##layers.append(nn.BatchNorm2d(16))
        #layers.append(nn.SELU())
        layers.append(MultiScaleFeature(8, 8, dilatation=2))
        layers.append(MultiScaleFeature(8, 8, dilatation=1))

        #layers.append(nn.ConvTranspose2d(32, 16,  (6, 6), stride=2, padding=2))
        #layers.append(nn.BatchNorm2d(16))
        #layers.append(nn.ReLU())
        #layers.append(nn.Conv2d(16, 16, (5, 5), stride=1, padding=2))
        #layers.append(nn.BatchNorm2d(16))
        #layers.append(nn.ReLU())
        #layers.append(nn.Conv2d(16, 16, (5, 5), stride=1, padding=2))
        #layers.append(nn.BatchNorm2d(16))
        #layers.append(nn.ReLU())

        #layers.append(nn.Conv2d(32, 16, (3, 3), stride=1, padding=1))
        #layers.append(nn.ReLU())
        #layers.append(nn.Conv2d(32, 32, (3, 3), stride=1, padding=1))
        #layers.append(nn.ReLU())
        #layers.append(nn.Conv2d(32, 16, (3, 3), stride=1, padding=1))

        self.dec3 = nn.Sequential(*layers)

        layers = []


        layers.append(nn.Conv2d(8, 8, (3, 3), stride=1, padding=1))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(8, dim[0], (1, 1), stride=1, padding=0))
        layers.append(nn.Sigmoid())

        self.decoder_final = nn.Sequential(*layers)

    def forward(self, x):
        fe = self.conv1(x)
        fe = self.bn1(fe)
        fe = self.relu1(fe)
        fe = self.maxpool1(fe)
        fe = self.layer1(fe)
        fe = self.layer2(fe)

        encoded = self.encoder(fe)

        upsampled1 = self.up1(encoded)
        decoded1 = self.dec1(encoded)

        upsampled2 = self.up2(upsampled1)
        decoded2 = self.dec2(decoded1)

        upsampled3 = self.up3(upsampled2)
        decoded3 = self.dec3(decoded2) * torch.sum(self.tan(upsampled3), axis=1).unsqueeze(1)
        #decoded3 = self.dec3(decoded2) * upsampled3

        decoded = self.decoder_final(decoded3) # unsqueeze

        return decoded


class ViT_CNN_mask(nn.Module):
    def __init__(self, dim):
        super(ViT_CNN_mask, self).__init__()
        self.dim = dim
        self.dec_emb_dim = 64

        vit = vit_b_16(pretrained=True)

        self.image_size = vit.image_size
        self.hidden_dim = vit.hidden_dim
        self.patch_size = vit.patch_size

        self.conv_proj = vit.conv_proj

        self.class_token = vit.class_token
        self.encoder = vit.encoder
        for param in self.conv_proj.parameters():
            param.requires_grad = True
        for param in self.encoder.parameters():
            param.requires_grad = True


        layers = []

        layers.append(nn.Linear(self.hidden_dim, (self.patch_size**2)*self.dec_emb_dim))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Unflatten(-1, (self.patch_size**2, self.dec_emb_dim)))
        self.encoder1 = nn.Sequential(*layers)

        self.tanh = nn.SELU()

        layers = []

        layers.append(nn.Conv2d(self.dec_emb_dim, 32, (1, 1), stride=1))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(32, 32, (3, 3), stride=1, padding=1))
        layers.append(nn.SELU())
        #layers.append(nn.Conv2d(32, 32, (1, 1), stride=1))
        #layers.append(nn.SELU())

        layers.append(nn.Conv2d(32, 16, (1, 1), stride=1))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(16, 16, (3, 3), stride=1, padding=1))
        layers.append(nn.SELU())
        #layers.append(nn.Conv2d(16, 16, (1, 1), stride=1))
        #layers.append(nn.SELU())


        layers.append(nn.Conv2d(16, 8, (1, 1), stride=1))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(8, 8, (3, 3), stride=1, padding=1))
        layers.append(nn.SELU())

        self.decoder1 = nn.Sequential(*layers)

        layers = []

        #layers.append(nn.Conv2d(8, dim[0], (3, 3), stride=1, padding=1))
        #layers.append(nn.SELU())
        #layers.append(nn.Conv2d(8, 8, (3, 3), stride=1, dilation=2, padding=2))
        #layers.append(nn.SELU())
        #layers.append(nn.Conv2d(8, 8, (3, 3), stride=1, padding=1))
        #layers.append(nn.SELU())
        #layers.append(nn.Conv2d(8, 8, (3, 3), stride=1, dilation=2, padding=2))
        #layers.append(nn.SELU())
        #layers.append(nn.Conv2d(8, 8, (3, 3), stride=1, padding=1))
        #layers.append(nn.SELU())
        layers.append(nn.Conv2d(9, dim[0], (1, 1), stride=1))
        layers.append(nn.Sigmoid())

        self.decoder2 = nn.Sequential(*layers)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, "Wrong image height!")
        torch._assert(w == self.image_size, "Wrong image width!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x
    
    def forward(self, x):
        x = self._process_input(x)
        n = x.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        encoded = self.encoder(x)[:, 1:] # Eliminiamo il token di classe

        p = self.patch_size
        B, N, _ = encoded.shape
        dec_emb_dim = self.dec_emb_dim
        h = w = self.image_size
        n_h = h // p
        n_w = w // p

        encoded_square = torch.sqrt(torch.sum(torch.pow(encoded, 2), axis=-1)).unsqueeze(-1)  # B, N, 1
        encoded_square = encoded_square.expand(-1, -1, p ** 2).view(B, N, p, p)
        encoded_square = rearrange(encoded_square, 'b (nh nw) ph pw -> b (nh ph) (nw pw)', nh=n_h, nw=n_w)

        decoded = self.encoder1(encoded)#.view(-1, self.image_size, self.image_size, self.dec_emb_dim).permute(0, 3, 1, 2)
        B, N, _, _ = decoded.shape

        decoded1 = decoded.view(B, N, dec_emb_dim, p, p)
        decoded1 = rearrange(decoded1, 'b (nh nw) c ph pw -> b c (nh ph) (nw pw)', nh=n_h, nw=n_w)

        decoded = self.decoder1(decoded1) # unsqueeze
        decoded = self.decoder2(torch.cat([decoded, encoded_square.unsqueeze(1)], dim=1))

        return decoded

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(F.adaptive_avg_pool2d(x, 1))
        max_out = self.shared_MLP(F.adaptive_max_pool2d(x, 1))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out