import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from tools.evaluation_metrics import Xauc

if __name__=='__main__':

    htmap_stats = []
    det_stats = []

    for i in range(10):
        #ret_path_d_64 = os.path.join('results', 'params', '64_16', 'mnist', str(i), '2')
        #ret_path_d_128 = os.path.join('results', 'params', '128_16', 'mnist', str(i), '2')
        ret_path = os.path.join('results', 'mnist', str(i), '29')
        ret_path_conv = os.path.join('/media/rosariaspada/KUBUNTU 22_/f_1/mnist', str(i), '29')
        GT_test = np.load(open(os.path.join(ret_path_conv, 'gt.npy'), 'rb'))
        Y_test = np.load(open(os.path.join(ret_path_conv, 'labels.npy'), 'rb'))

        # AE-XAD conv
        htmaps_aexad_conv = np.load(open(os.path.join(ret_path_conv, 'aexad_htmaps_conv.npy'), 'rb'))
        scores_aexad_conv = np.load(open(os.path.join(ret_path_conv, 'aexad_scores_conv.npy'), 'rb'))
        # AE conv
        htmaps_ae = np.load(open(os.path.join(ret_path, 'ae_htmaps.npy'), 'rb'))
        scores_ae = np.load(open(os.path.join(ret_path, 'ae_scores.npy'), 'rb'))
        # Deviation
        htmaps_dev = np.load(open(os.path.join(ret_path, 'deviation_htmaps.npy'), 'rb'))
        scores_dev = np.load(open(os.path.join(ret_path, 'deviation_scores.npy'), 'rb'))
        # FCDD
        htmaps_fcdd = np.load(open(os.path.join(ret_path, 'fcdd_htmaps.npy'), 'rb'))
        scores_fcdd = np.load(open(os.path.join(ret_path, 'fcdd_scores.npy'), 'rb'))

        htmap_stats.append([Xauc(GT_test[Y_test == 1], htmaps_aexad_conv[Y_test == 1]),
                            Xauc(GT_test[Y_test == 1], htmaps_ae[Y_test == 1]),
                            Xauc(GT_test[Y_test == 1], htmaps_dev[Y_test == 1]),
                            Xauc(GT_test[Y_test == 1], htmaps_fcdd[Y_test == 1])])
        det_stats.append([roc_auc_score(Y_test, scores_aexad_conv),
                          roc_auc_score(Y_test, scores_ae),
                          roc_auc_score(Y_test, scores_dev),
                          roc_auc_score(Y_test, scores_fcdd)])
    htmap_stats = np.array(htmap_stats).reshape((10, 4))
    det_stats = np.array(det_stats).reshape((10, 4))

    # Only detection
    plt.figure(1)
    plt.title('Detection')
    plt.plot(det_stats[:, 0], label='AE-XAD conv')
    plt.plot(det_stats[:, 1], label='AE conv')
    plt.plot(det_stats[:, 2], label='deviation')
    plt.plot(det_stats[:, 3], label='FCDD')
    plt.legend()
    plt.ylim(0.5, 1.001)
    plt.show()

    # Only explanation
    plt.figure(2)
    plt.title('Explanation')
    plt.plot(htmap_stats[:, 0], label='AE-XAD conv')
    plt.plot(htmap_stats[:, 1], label='AE conv')
    plt.plot(htmap_stats[:, 2], label='deviation')
    plt.plot(htmap_stats[:, 3], label='FCDD')
    plt.legend()
    plt.ylim(0.5, 1.001)
    plt.show()
