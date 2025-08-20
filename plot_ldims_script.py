import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from tools.evaluation_metrics import Xauc

if __name__=='__main__':

    htmap_stats = []
    det_stats = []
    ldims = [8, 16, 32, 64, 128, 256]
    for ld in ldims:
        for i in range(10):
            ret_path = os.path.join('results', 'params', f'{ld}_16', 'mnist', str(i), '2')
            GT_test = np.load(open(os.path.join(ret_path, 'gt.npy'), 'rb'))
            Y_test = np.load(open(os.path.join(ret_path, 'labels.npy'), 'rb'))
            htmaps_aexad = np.load(open(os.path.join(ret_path, 'aexad_htmaps.npy'), 'rb'))
            scores_aexad = np.load(open(os.path.join(ret_path, 'aexad_scores.npy'), 'rb'))
            htmap_stats.append(Xauc(GT_test[Y_test == 1], htmaps_aexad[Y_test == 1]))
            det_stats.append(roc_auc_score(Y_test, scores_aexad))
    htmap_stats = np.array(htmap_stats).reshape((len(ldims), 10))
    det_stats = np.array(det_stats).reshape((len(ldims), 10))

    # Only detection
    plt.title('Detection')
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.plot(det_stats[:, i])
        plt.ylim(0.0, 1.1)
        plt.xticks(range(len(ldims)), ldims)
    plt.show()

    # Only explanation
    plt.title('Detection')
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.plot(det_stats[:, i])
        plt.ylim(0.0, 1.1)
        plt.xticks(range(len(ldims)), ldims)
    plt.show()

    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.plot(det_stats[:, i], label='Detection')
        plt.plot(htmap_stats[:, i], label='Explanation')
        plt.legend()
        plt.ylim(0.5, 1.001)
        plt.xticks(range(len(ldims)), ldims)
    plt.show()