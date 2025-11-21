import torch
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.ndimage import binary_closing, gaussian_filter


def gaussian_smoothing(hm, kernel_size=21, sigma=4.0):
    """
    hm: numpy array (B,H,W) o torch tensor (B,H,W)
    output: numpy array (B,H,W)
    """
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


def refine_sharp_anomaly(hm, blur_ks=21, blur_sigma=8):
    """
    Refinement per anomalie sottili:
    - enfatizza local contrast
    - preserva bordi stretti
    - non gonfia la heatmap
    """
    hm = hm.astype(np.float32)

    # local blur
    blurred = cv2.GaussianBlur(hm, (blur_ks, blur_ks), blur_sigma)

    # local contrast enhancement
    refined = hm - blurred
    refined = np.maximum(refined, 0)

    return refined


def heatmap_refine(hm, sigma=1.0, bilateral_d=9, bilateral_sigma_color=30, bilateral_sigma_space=7):
    """
    Migliora la heatmap in tre fasi, senza alterarne la struttura:
    1) Bilateral filtering (preserva i bordi)
    2) Morphological closing (chiude buchi)
    3) Patch smoothing finale
    """

    # -> numpy float32
    hm = hm.astype(np.float32)

    # 1) Bilateral Filtering (edge-preserving)
    hm_bil = cv2.bilateralFilter(
        hm, 
        d=bilateral_d, 
        sigmaColor=bilateral_sigma_color,
        sigmaSpace=bilateral_sigma_space
    )

    # 2) Morphological Closing
    mask_bin = (hm_bil > (hm_bil.mean() + 0.5 * hm_bil.std())).astype(np.uint8)
    mask_closed = binary_closing(mask_bin, structure=np.ones((5,5))).astype(np.float32)

    hm_closed = hm_bil * 0.7 + mask_closed * 0.3

    # 3) Patch smoothing (leggero gaussian sulle linee delle patch)
    hm_final = gaussian_filter(hm_closed, sigma=sigma)

    return hm_final

def unsharp_mask(hm, amount=1.0, sigma=1.0):
    # gaussian blur piccolo
    blur = gaussian_smoothing(hm, kernel_size=5, sigma=sigma)
    # sharpen: hm + amount*(hm - blur)
    return hm + amount * (hm - blur)


import torch
import torch.nn.functional as F

def refine_hmap(hmap_tensor, kernel_size=7, sigma=2.0):
    """
    Refinement non-learnable della heatmap.
    Input:  tensor shape (B, H, W)
    Output: tensor shape (B, H, W) refinito
    """
    if isinstance(hmap_tensor, np.ndarray):
        h = torch.tensor(hmap_tensor, dtype=torch.float32)
    else:
        h = hmap_tensor.clone().float()

    # ---- 1) Gaussian smoothing leggero ----
    h = h.unsqueeze(1)                      # (B,1,H,W)
    h = gaussian_blur(h, kernel_size, sigma)
    
    # ---- 2) Local contrast normalization ----
    local_mean = F.avg_pool2d(h, kernel_size=9, stride=1, padding=4)
    local_var  = F.avg_pool2d((h - local_mean)**2, kernel_size=9, stride=1, padding=4)
    local_std  = torch.sqrt(local_var + 1e-6)
    h = (h - local_mean) / (local_std + 1e-6)

    # ---- 3) Unsharp masking (molto leggero) ----
    blur = gaussian_blur(h, kernel_size=9, sigma=3.0)
    h = h + 0.25 * (h - blur)

    # ---- 4) Clamp 0–1 ----
    h = h.squeeze(1)
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)
    h = torch.clamp(h, 0, 1)

    return h.numpy()

def gaussian_blur(x, kernel_size=7, sigma=2.0):
    from torchvision.transforms.functional import gaussian_blur as gb
    return gb(x, kernel_size=kernel_size, sigma=sigma)
