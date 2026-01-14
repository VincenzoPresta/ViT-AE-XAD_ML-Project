import torch
import torch.nn.functional as F
import numpy as np


def gaussian_smoothing(hm, kernel_size=21, sigma=4.0):
    
    if isinstance(hm, np.ndarray):
        hm = torch.from_numpy(hm).float()

    B, H, W = hm.shape

    coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    xx = coords.repeat(kernel_size).view(kernel_size, kernel_size)
    kernel = torch.exp(-(xx**2 + xx.T**2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()

    kernel = kernel.view(1, 1, kernel_size, kernel_size).to(hm.device)

    hm = hm.unsqueeze(1)  # (B,1,H,W)
    hm = F.conv2d(hm, kernel, padding=kernel_size // 2)
    hm = hm.squeeze(1)

    return hm.detach().cpu().numpy()