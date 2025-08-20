import numpy as np
import os

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from PIL import Image

from tools.evaluation_metrics import Xauc
from tools.create_dataset import mvtec, mvtec_personalized
from tools.load_data import load_data_aexad, load_data_fcdd, load_data_devnet
from tools.utils import plot_data, blurred_htmaps

import warnings
warnings.filterwarnings('ignore')

object_names = (
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
    'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
    'wood', 'zipper'
)


def plot_htmap(ht_id, hts, gts, imgs):
    plt.subplot(1,3,1)
    plt.imshow(imgs[ht_id])
    plt.subplot(1, 3, 2)
    plt.imshow(hts[ht_id])
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(gts[ht_id])
    plt.show()

if __name__ == '__main__':
    #c = 1
    at = 0
    na = 3
    s = 40

    for c in range(10, 11):
        root = os.path.join('results', 'mvtec_our')

        ht_aexad, sc_aexad, imgs_aexad, labels, gt_aexad, _ = load_data_aexad(root, c, na, s)
        ht_fcdd, sc_fcdd, imgs_fcdd, _, gt_fcdd = load_data_fcdd(root, c, na, s)
        ht_deviation, sc_deviation, imgs_deviation, _, gt_deviation = load_data_devnet(root, c, na, s)

        ht_aexad_blur = blurred_htmaps(ht_aexad, scale=1.0)
        ht_fabrizio = ht_aexad * ht_aexad_blur
        sc_fabrizio = ht_fabrizio.sum(axis=(2,1))/(448*448)

        # ------------------ Compute metrics ---------------------------------
        ht_auc_aexad = np.empty(gt_aexad[labels == 1].shape[0])
        ht_auc_aexad_blur = np.empty(gt_aexad[labels == 1].shape[0])
        ht_auc_fabrizio = np.empty(gt_aexad[labels == 1].shape[0])
        ht_auc_fcdd = np.empty(gt_fcdd[labels==1].shape[0])
        ht_auc_deviation = np.empty(gt_deviation[labels==1].shape[0])
        for i in range(gt_aexad[labels == 1].shape[0]):
            ht_auc_aexad_blur[i] = Xauc(gt_aexad[labels == 1][i], ht_aexad_blur[labels == 1][i])
            ht_auc_aexad[i] = Xauc(gt_aexad[labels == 1][i], ht_aexad[labels == 1][i])
            ht_auc_fabrizio[i] = Xauc(gt_aexad[labels == 1][i], ht_fabrizio[labels == 1][i])
            ht_auc_fcdd[i] = Xauc(gt_fcdd[labels==1][i], ht_fcdd[labels==1][i])
            ht_auc_deviation[i] = Xauc(gt_deviation[labels==1][i], ht_deviation[labels==1][i])
        sc_auc_aexad = roc_auc_score(labels, sc_aexad)
        sc_auc_fabrizio = roc_auc_score(labels, sc_fabrizio)
        sc_auc_fcdd = roc_auc_score(labels, sc_fcdd)
        sc_auc_deviation = roc_auc_score(labels, sc_deviation)

        #print(f'----- NORM CLASS {c} -----')
        #print('-- Detection --')
        #print('FCDD:', sc_auc_fcdd)
        #print('DEVIATION', sc_auc_deviation)
        #print('AE-XAD', sc_auc_aexad)
        #print('-- Explanation --')
        #print('FCDD:', ht_auc_fcdd.mean())
        #print('DEVIATION', ht_auc_deviation.mean())
        #print('AE-XAD', ht_auc_aexad_blur.mean())
        #print('AE-XAD NO FILTER', ht_auc_aexad.mean())
        print(object_names[c], end=' & ')
        # Detection
        print(np.round(sc_auc_fcdd, 4), end=' & ')
        print(np.round(sc_auc_deviation, 4), end=' & ')
        print(np.round(sc_auc_aexad, 4), end=' & ')
        print(np.round(sc_auc_fabrizio, 4), end=' & ')
        # Explanation
        print(np.round(ht_auc_fcdd.mean(), 4), end=' & ')
        print(np.round(ht_auc_deviation.mean(), 4), end=' & ')
        print(np.round(ht_auc_aexad_blur.mean(), 4), end='  & ')
        print(np.round(ht_auc_fabrizio.mean(), 4), end=' \\\\\n')
