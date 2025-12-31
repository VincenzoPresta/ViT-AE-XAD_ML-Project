# testing_tools.py

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label


# ============================================
# NORMALIZZAZIONE (paper)
# ============================================

def normalize_error(img, out, mode="2"):
    """
    Return e_tilde as 2D map (H,W), consistent with paper heatmap pipeline.
    e_tilde = sum_c ( (img-out)^2 / ((F_t - img)^2 + eps) )
    """
    if mode == "2":
        F_t = 2.0
    elif mode == "1-x":
        F_t = 1.0 - img
    else:
        raise ValueError("Unknown normalization mode")

    num = (img - out) ** 2           # (3,H,W)
    den = (F_t - img) ** 2 + 1e-8    # (3,H,W) if F_t scalar broadcasts, OK
    e_tilde_c = num / den            # (3,H,W)

    # collapse channels -> (H,W)
    return e_tilde_c.sum(dim=0)



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

def compute_score(e, e_filt):
    """
    Paper: S(t) = || e · F_k(e) ||.
    Use L2 norm (Frobenius) on the 2D map.
    """
    m = e * e_filt
    return float(torch.norm(m, p=2).detach().cpu().numpy())



def aexad_heatmap_and_score(img_np, out_np, label):
    img_t = torch.tensor(img_np, dtype=torch.float32)  # (3,H,W) oppure (H,W,3) ? assumo già (3,H,W)
    out_t = torch.tensor(out_np, dtype=torch.float32)

    # se arrivano in HWC, normalizza qui
    if img_t.ndim == 3 and img_t.shape[0] != 3 and img_t.shape[-1] == 3:
        img_t = img_t.permute(2, 0, 1)
        out_t = out_t.permute(2, 0, 1)

    # 1) raw error e (H,W)
    e = ((img_t - out_t) ** 2).sum(dim=0)

    # 2) normalized error e_tilde (H,W) - Eq.(3)
    e_tilde = normalize_error(img_t, out_t, mode="2")

    # 3) k-hat from e_tilde
    e_tilde_np = np.squeeze(e_tilde.detach().cpu().numpy())
    k_hat = estimate_k(e_tilde_np)

    # 4) score uses F_k(e) (NOT e_tilde)
    e_filt = gaussian_filter_paper(e, k_hat)
    score = compute_score(e, e_filt)

    # 5) heatmap uses F_k(e_tilde)
    h = gaussian_filter_paper(e_tilde, k_hat)
    h_np = h.detach().cpu().numpy()

    # 6) binarization on heatmap
    mu_h = h_np.mean()
    sigma_h = h_np.std()
    binary_h = (h_np > (mu_h + sigma_h)).astype(np.uint8)

    # --- DEBUG ---
    print("\n[DEBUG HEATMAP PAPER-COMPAT]")
    '''if label is not None:
        print("LABEL:", int(label))'''
    print("e (raw) min/max:", float(e.min()), float(e.max()), "mean/std:", float(e.mean()), float(e.std()))
    print("e_tilde (norm) min/max:", float(e_tilde.min()), float(e_tilde.max()), "mean/std:", float(e_tilde.mean()), float(e_tilde.std()))
    print("k_hat:", k_hat)
    print("e_filt (for score) min/max:", float(e_filt.min()), float(e_filt.max()), "mean/std:", float(e_filt.mean()), float(e_filt.std()))
    print("h (heatmap) min/max:", float(h.min()), float(h.max()), "mean/std:", float(h.mean()), float(h.std()))
    print("mu_h:", float(mu_h), "sigma_h:", float(sigma_h), "thr:", float(mu_h + sigma_h))
    print("binary unique:", np.unique(binary_h)[:10], "ratio_ones:", float(binary_h.mean()))

    return e.detach().cpu().numpy(), h_np, binary_h, score, k_hat
