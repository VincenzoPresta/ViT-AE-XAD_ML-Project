import numpy as np
import cv2
from scipy.ndimage import binary_closing, gaussian_filter

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