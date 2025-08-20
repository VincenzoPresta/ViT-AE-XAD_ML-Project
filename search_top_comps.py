import argparse
import os
import torch

import matplotlib.pyplot as plt
import numpy as np
from kornia import gaussian_blur2d

from AE_architectures import Conv_Autoencoder
from tools.create_dataset import square
from tools.evaluation_metrics import Xauc
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', type=str, help='Dataset to use')
    parser.add_argument('-s', type=int, nargs='+', help='Seeds')
    args = parser.parse_args()

    ds = args.ds

    # DEV, FCDD, AE, AE-XAD
    exp_aucs = np.full((10), fill_value=-5)
    best_imgs = np.empty((10, 6, 28, 28), dtype=np.float64)

    for i in range(10):
        for ns in range(len(args.s)):
            s = args.s[ns]

            ret_path_aexad = os.path.join('results', 'f_1', args.ds, str(i), str(s))
            ret_path_ae = os.path.join('results', args.ds, str(i), str(s))
            ret_path_comp = os.path.join('results', f'{args.ds}_comp', str(i), str(s))

            _, _, X_test, Y_test, _, GT_test = \
                square(i, perc_anom_train=0.02, perc_anom_test=0.1, size=5,
                       intensity='rand', DATASET=ds, seed=s)

            htmaps_aexad = np.load(open(os.path.join(ret_path_aexad, 'aexad_htmaps_conv.npy'), 'rb'))
            htmaps_aexad_f = gaussian_blur2d(torch.from_numpy(htmaps_aexad), kernel_size=(9, 9),
                                       sigma=(2, 2))
            htmaps_ae = np.load(open(os.path.join(ret_path_ae, 'ae_htmaps.npy'), 'rb'))
            htmaps_dev = np.load(open(os.path.join(ret_path_comp, 'deviation_htmaps.npy'), 'rb'))
            htmaps_fcdd = np.load(open(os.path.join(ret_path_comp, 'fcdd_htmaps.npy'), 'rb'))

            exps = np.empty((len(GT_test[Y_test == 1])), dtype=np.float64)
            for j in range(len(GT_test[Y_test == 1])):
                exps[j] = Xauc(GT_test[Y_test == 1][j], htmaps_aexad_f[Y_test == 1][j]) - \
                          Xauc(GT_test[Y_test == 1][j], htmaps_fcdd[Y_test == 1][j])

            if exps.max() > exp_aucs[i]:
                id = np.argmax(exps)
                exp_aucs[i] = exps.max()
                best_imgs[i, 0] = X_test[Y_test == 1][id]
                best_imgs[i, 1] = GT_test[Y_test == 1][id]
                best_imgs[i, 2] = htmaps_aexad_f[Y_test == 1][id]
                best_imgs[i, 3] = htmaps_ae[Y_test == 1][id]
                best_imgs[i, 5] = htmaps_fcdd[Y_test == 1][id]
                best_imgs[i, 4] = htmaps_dev[Y_test == 1][id]

    np.save(open(f'{args.ds}_best_imgs.npy', 'wb'), best_imgs)

    classes = ['T-shirt\ \ntop',
               'Trouser',
               'Pullover',
               'Dress',
               'Coat',
               'Sandal',
               'Shirt',
               'Sneaker',
               'Bag',
               'Ankle\nboot']
    fig, axs = plt.subplots(6, 10, sharex=True, sharey=True, figsize=(6, 3.9))
    for i in range(10):
        axs[0, (i % 10)].imshow(best_imgs[i, 0], cmap='gray')
        axs[0, (i % 10)].set_title(classes[i], fontdict={'fontsize': 'x-small'})
        axs[1, (i % 10)].imshow(best_imgs[i, 1], cmap='gray')
        axs[2, (i % 10)].imshow(best_imgs[i, 2], cmap='jet')
        axs[3, (i % 10)].imshow(best_imgs[i, 3], cmap='jet')
        axs[4, (i % 10)].imshow(best_imgs[i, 4], cmap='jet')
        axs[5, (i % 10)].imshow(best_imgs[i, 5], cmap='jet')
        plt.setp(axs[0, (i % 10)].get_xticklabels(), visible=False)
        plt.setp(axs[0, (i % 10)].get_yticklabels(), visible=False)
        plt.setp(axs[1, (i % 10)].get_xticklabels(), visible=False)
        plt.setp(axs[1, (i % 10)].get_yticklabels(), visible=False)
        plt.setp(axs[2, (i % 10)].get_xticklabels(), visible=False)
        plt.setp(axs[2, (i % 10)].get_yticklabels(), visible=False)
        plt.setp(axs[3, (i % 10)].get_xticklabels(), visible=False)
        plt.setp(axs[3, (i % 10)].get_yticklabels(), visible=False)
        plt.setp(axs[4, (i % 10)].get_xticklabels(), visible=False)
        plt.setp(axs[4, (i % 10)].get_yticklabels(), visible=False)
        plt.setp(axs[5, (i % 10)].get_xticklabels(), visible=False)
        plt.setp(axs[5, (i % 10)].get_yticklabels(), visible=False)
        axs[0, (i % 10)].set_xticks([])
        axs[1, (i % 10)].set_xticks([])
        axs[2, (i % 10)].set_xticks([])
        axs[3, (i % 10)].set_xticks([])
        axs[4, (i % 10)].set_xticks([])
        axs[5, (i % 10)].set_xticks([])
        if i == 0:
            axs[0, 0].set(ylabel='IMAGE')
            axs[1, 0].set(ylabel='GT')
            axs[2, 0].set(ylabel='AE-XAD')
            axs[3, 0].set(ylabel='AE')
            axs[4, 0].set(ylabel='FCDD')
            axs[5, 0].set(ylabel='DevNet')
        axs[0, (i % 10)].tick_params(axis='both', which='both', length=0)
        axs[1, (i % 10)].tick_params(axis='both', which='both', length=0)
        axs[2, (i % 10)].tick_params(axis='both', which='both', length=0)
        axs[3, (i % 10)].tick_params(axis='both', which='both', length=0)
        axs[4, (i % 10)].tick_params(axis='both', which='both', length=0)
        axs[5, (i % 10)].tick_params(axis='both', which='both', length=0)
    plt.subplots_adjust(wspace=0, hspace=0., left=0.05, top=0.92, bottom=0.02, right=1.0)
    plt.savefig('fmnist.eps', bbox_inches="tight")
    plt.show()


