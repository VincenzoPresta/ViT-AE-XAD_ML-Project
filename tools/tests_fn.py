import numpy as np
from tools.utils import plot_data, get_filter_sigma
from tools.evaluation_metrics import Xauc
from scipy.ndimage import gaussian_filter


def test_adaptive_mask(images, gts, labels, htmaps, scales):
    # Assumiamo che htmpas contenga la media per ogni pixel
    for s in scales:
        htmaps_f = np.empty_like(htmaps[labels==1])
        exp_auc = np.empty(htmaps[labels==1].shape[0])
        for i in range(htmaps[labels==1].shape[0]):
            sigma = get_filter_sigma(htmaps[labels==1][i], scale=s)
            htmaps_f[i] = gaussian_filter(htmaps[labels==1][i], sigma=sigma, truncate=3)
            exp_auc[i] = Xauc(gts[labels==1][i], htmaps_f[i])
        print(f'SCALE {s} EXP AUC {exp_auc.mean()}')