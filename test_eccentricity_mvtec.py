from torchvision import transforms
from PIL import Image

from tools.create_dataset import mvtec_ac, mvtec_personalized, mvtec_all_classes
from tools.utils import plot_data, plot_htmaps, blurred_htmaps
from tools.load_data import load_model, load_data_fcdd
import os
import numpy as np

from tools.load_data import load_data_aexad
from tools.load_data import mvtec
from tools.evaluation_metrics import Xauc
from sklearn.metrics import roc_auc_score

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate

import pickle

classes_names = (
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper'
    )

if __name__ == '__main__':
    na = 1
    s = 40

    ecc_info = []
    xauc_aexad_info = []
    xauc_fcdd_info = []
    auc_aexad_info = []
    auc_fcdd_info = []

    results_path = f'/media/rosariaspada/Dati/AE-XAD-v2/results/mvtec_all'

    for c in range(15):
        print(f'-------------- {classes_names[c]} ----------------')
        #X_train, Y_train, X_test, Y_test, GT_train, GT_test, anomaly_types = mvtec_ours(c, 'datasets/mvtec', n_anom_per_cls=na, seed=s)
        #X_train, Y_train, X_test, Y_test, GT_train, GT_test, anomaly_types, outlier_classes, files_train, files_test = mvtec_(c,
        #                                                                                                        'datasets/mvtec',
        #                                                                                                                                  n_anom_per_cls=na,
        #                                                                                                            seed=s)
        X_train, Y_train, X_test, Y_test, GT_train, GT_test, anomaly_types, outlier_classes, files_train, files_test = mvtec_all_classes(c,
                                                                                                               'datasets/mvtec',
                                                                                                               n_anom_per_cls=na,
                                                                                                               seed=s)
        #print(X_test.shape)
        #print(outlier_classes)
        #print(anomaly_types.shape)
        #print(anomaly_types)

        results_aexad = f'/media/rosariaspada/Dati/AE-XAD-v2/results/mvtec_all/{c}/{s}/{na}'
        results_fcdd = f'/media/rosariaspada/Dati/AE-XAD-v2/results/mvtec_all/'
        #results_aexad = results_fcdd

        ht_aexad_res, sc_aexad, _ = load_data_aexad(results_aexad, 'aexad_htmaps_conv_deep_v2.npy', 'aexad_scores_conv_deep_v2.npy',
                                                    shape=GT_test.shape[:3])
        ht_aexad_blur = blurred_htmaps(ht_aexad_res, scale=1.0)
        ht_fabrizio = ht_aexad_res * ht_aexad_blur
        sc_fabrizio = ht_fabrizio.sum(axis=(2, 1)) / (448 * 448)
        ht_fcdd_res, sc_fcdd = load_data_fcdd(results_fcdd, c, na, s, shape=GT_test.shape[:3])

        ecc_info.append([])
        xauc_aexad_info.append([])
        xauc_fcdd_info.append([])


        for at in outlier_classes:
            #an_gt = GT_test[Y_test == 1, :, :, 0][anomaly_types == at]
            an_gt = GT_test[:, :, :, 0][anomaly_types == at]
            an_ht_fcdd = ht_fcdd_res[anomaly_types == at]
            an_ht_aexad = ht_aexad_blur[anomaly_types == at]
            an_data = X_test[anomaly_types == at]

            labels = np.empty(an_gt.shape, dtype=int)
            regions = np.empty(an_gt.shape[0], dtype=object)
            eccenticities = np.empty(an_gt.shape[0], dtype=np.float32)
            for i in range(an_gt.shape[0]):
                labels[i] = label(an_gt[i])
                regions[i] = regionprops(labels[i])
                nr = len(regions[i])
                eccs = np.empty(nr)
                pixels = np.empty(nr)
                for j in range(nr):
                    eccs[j] = regions[i][j].eccentricity
                    pixels[j] = np.where(labels[i] == (j + 1), 1, 0).sum()
                eccenticities[i] = (eccs * pixels).sum() / pixels.sum()

            xaucs_aexad = np.empty(an_gt.shape[0])
            xaucs_fcdd = np.empty(an_gt.shape[0])
            for i in range(an_gt.shape[0]):
                xaucs_aexad[i] = Xauc(an_gt[i], an_ht_aexad[i])
                xaucs_fcdd[i] = Xauc(an_gt[i], an_ht_fcdd[i])

            ecc_info[c].append([eccenticities.mean(), eccenticities.std()])
            xauc_aexad_info[c].append([xaucs_aexad.mean(), xaucs_aexad.std()])
            xauc_fcdd_info[c].append([xaucs_fcdd.mean(), xaucs_fcdd.std()])
        auc_aexad_info.append(roc_auc_score(Y_test, sc_fabrizio))
        auc_fcdd_info.append(roc_auc_score(Y_test, sc_fcdd))


        pickle.dump(ecc_info, open(f'{results_path}/ecc_info.pkl', 'wb'))
        pickle.dump(xauc_aexad_info, open(f'{results_path}/xauc_aexad_info.pkl', 'wb'))
        pickle.dump(xauc_fcdd_info, open(f'{results_path}/xauc_fcdd_info.pkl', 'wb'))
        pickle.dump(auc_aexad_info, open(f'{results_path}/auc_aexad_info.pkl', 'wb'))
        pickle.dump(auc_fcdd_info, open(f'{results_path}/auc_fcdd_info.pkl', 'wb'))

        print('ECCENTRICITY', ecc_info[c], 'AEXAD XAUC', xauc_aexad_info[c], 'FCDD XAUC', xauc_fcdd_info[c],
              'AEXAD AUC', auc_aexad_info[c], 'FCDD AUC', auc_fcdd_info[c], sep='\n')