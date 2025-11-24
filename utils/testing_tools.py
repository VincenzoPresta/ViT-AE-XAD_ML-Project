# testing_tools.py

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label


# ============================================
# NORMALIZZAZIONE (paper)
# ============================================

def normalize_error(e, img, out, mode="2"):
    if mode == "2":
        F_t = 2.0
    elif mode == "1-x":
        F_t = 1 - img
    else:
        raise ValueError("Unknown normalization mode")

    num = (img - out)**2
    den = (F_t - img)**2 + 1e-8
    return num / den


# ============================================
# STIMA AUTOMATICA DI k̂ (paper)
# ============================================

def estimate_k(e_tilde):
    # ensure 2D
    e_tilde = np.squeeze(e_tilde)

    mu = e_tilde.mean()
    sigma = e_tilde.std()

    bw = (e_tilde > (mu + sigma)).astype(np.uint8)

    # ensure 2D
    bw = np.squeeze(bw)

    if bw.ndim != 2:
        # fallback: collapse last dimension
        bw = bw[..., 0]

    if bw.max() == 0:
        return 1

    labeled, num = label(bw)

    lengths_h = []
    lengths_v = []

    for comp in range(1, num + 1):
        # now ALWAYS yields ys,xs only
        ys, xs = np.where(labeled == comp)

        if len(xs) > 0:
            lengths_h.append(xs.max() - xs.min() + 1)
        if len(ys) > 0:
            lengths_v.append(ys.max() - ys.min() + 1)

    if not lengths_h or not lengths_v:
        return 1

    L = max(np.mean(lengths_h), np.mean(lengths_v))
    return max(1, int(L / 2))



# ============================================
# GAUSSIAN FILTER (paper)
# ============================================

def gaussian_filter_paper(e_tilde, k_hat):
    kernel_size = 2 * k_hat + 1
    sigma = k_hat / 3

    coords = torch.arange(kernel_size).float() - k_hat
    xx = coords.repeat(kernel_size).view(kernel_size, kernel_size)
    kernel = torch.exp(-(xx**2 + xx.T**2) / (2 * sigma * sigma))
    kernel /= kernel.sum()

    kernel = kernel.view(1,1,kernel_size,kernel_size).to(e_tilde.device)

    et = e_tilde.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    h = F.conv2d(et, kernel, padding=k_hat).squeeze(0).squeeze(0)

    return h


# ============================================
# SCORE S(t) = || e * h ||
# ============================================

def compute_score(e, filtered):
    return float((e * filtered).sum().cpu().numpy())


# ============================================
# AE-XAD PIPELINE COMPLETA (paper)
# ============================================

def aexad_heatmap_and_score(img_np, out_np):
    img_t = torch.tensor(img_np, dtype=torch.float32)
    out_t = torch.tensor(out_np, dtype=torch.float32)

    # RAW error
    e = ((img_t - out_t)**2).sum(dim=0)

    # Normalized error
    e_tilde = normalize_error(e, img_t, out_t, mode="2")

    # FIX: collapse channels if any (paper expects a 2D matrix)
    if e_tilde.ndim > 2:
        e_tilde = e_tilde.mean(dim=0)   # (3,H,W) -> (H,W)

    # k̂ estimation
    e_tilde_np = e_tilde.cpu().numpy()
    e_tilde_np = np.squeeze(e_tilde_np)  # garantisce (H,W)
    k_hat = estimate_k(e_tilde_np)


    # Filtered map
    filtered = gaussian_filter_paper(e_tilde, k_hat)
    filtered_np = filtered.cpu().numpy()

    # Binarization (paper)
    mu_h = filtered_np.mean()
    sigma_h = filtered_np.std()
    binary_h = (filtered_np > (mu_h + sigma_h)).astype(np.uint8)

    # Score
    score = compute_score(e, filtered)

    return e.numpy(), filtered_np, binary_h, score, k_hat
