# testing_tools.py

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label


# ============================================
#           NORMALIZZAZIONE (paper)
# ============================================


def normalize_error(e, img, out):
    F_t = 2.0

    # numerator: sum over channels
    num = ((img - out) ** 2).sum(dim=0)  # (H,W)
    # denominator: also pixelwise, sum channels
    den = ((F_t - img) ** 2).sum(dim=0) + 1e-8  # (H,W)

    return num / den  # (H,W)


# ============================================
#         STIMA AUTOMATICA DI k̂ (paper)
# ============================================


def estimate_k(e_tilde):
    mu = e_tilde.mean()
    sigma = e_tilde.std()
    bw = (e_tilde > (mu + sigma)).astype(np.uint8)

    if bw.max() == 0:
        return 1

    labeled, num = label(bw)
    lengths_h = []
    lengths_v = []

    for comp in range(1, num + 1):
        ys, xs = np.where(labeled == comp)
        if len(xs) > 0:
            lengths_h.append(xs.max() - xs.min() + 1)
        if len(ys) > 0:
            lengths_v.append(ys.max() - ys.min() + 1)

    if len(lengths_h) == 0 or len(lengths_v) == 0:
        return 1

    L = max(np.mean(lengths_h), np.mean(lengths_v))
    return max(1, int(L / 2))


# ============================================
#        FILTRO GAUSSIANO (paper)
# ============================================


def gaussian_filter_paper(e_tilde, k_hat):
    kernel_size = 2 * k_hat + 1
    sigma = k_hat / 3

    coords = torch.arange(kernel_size).float() - k_hat
    xx = coords.repeat(kernel_size).view(kernel_size, kernel_size)
    kernel = torch.exp(-(xx**2 + xx.T**2) / (2 * sigma * sigma))
    kernel /= kernel.sum()

    kernel = kernel.view(1, 1, kernel_size, kernel_size).to(e_tilde.device)

    et = e_tilde.unsqueeze(0).unsqueeze(0)
    h = F.conv2d(et, kernel, padding=k_hat).squeeze()

    return h


# ============================================
#           SCORE S(t) = || e · Fk(e) ||
# ============================================


def compute_score(e, filtered):
    return float((e * filtered).sum())


# ============================================
#      PIPELINE COMPLETA AE-XAD (paper)
# ============================================


def aexad_heatmap_and_score(img_np, out_np):
    img_t = torch.tensor(img_np, dtype=torch.float32)
    out_t = torch.tensor(out_np, dtype=torch.float32)

    # raw error
    e = ((img_t - out_t) ** 2).sum(dim=0)

    # normalized error (paper)
    e_tilde = normalize_error(e, img_t, out_t)

    # k̂ estimation
    # ensure 2D shape (H,W)
    e_tilde_np = e_tilde.cpu().numpy()
    while e_tilde_np.ndim > 2:
        e_tilde_np = e_tilde_np[0]  # elimina dimensioni extra create da pytorch

    k_hat = estimate_k(e_tilde_np)

    # filtered map
    filtered = gaussian_filter_paper(e_tilde, k_hat)
    filtered_np = filtered.cpu().numpy()

    # === NORMALIZZAZIONE MIN-MAX (paper) ===
    h_norm = (filtered_np - filtered_np.min()) / (
        filtered_np.max() - filtered_np.min() + 1e-8
    )

    # binarization µh + σh
    mu_h = h_norm.mean()
    sigma_h = h_norm.std()
    binary_h = (h_norm > (mu_h + sigma_h)).astype(np.uint8)

    # score S(t)
    score = compute_score(e, filtered)

    return e.numpy(), filtered_np, binary_h, score, k_hat
