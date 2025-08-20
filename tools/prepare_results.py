import os

from torchvision import transforms
from PIL import Image

from tools.create_dataset import mvtec_ac, mvtec_personalized, mvtec_all_classes
from tools.utils import plot_data, plot_htmaps, blurred_htmaps
from tools.load_data import load_model, load_data_fcdd

from tools.load_data import load_data_aexad
from tools.load_data import mvtec
from tools.evaluation_metrics import Xauc, average_precision
from sklearn.metrics import roc_auc_score
import numpy as np
import warnings

warnings.filterwarnings("ignore")


classes_names = (
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper'
    )

indexes = ( 0, 1, 2, 5, 7, 8, 9, 11, 12, 14, 3, 4, 6, 10, 13)


if __name__ == '__main__':
    na = 1#10
    s = 40
    dataset = 'mvtec'

    if dataset == 'mvtec':
        results_path = f'/media/rosariaspada/Dati/AE-XAD-v2/results/mvtec_all'

        for c in indexes:#range(15):
            X_train, Y_train, X_test, Y_test, GT_train, GT_test, anomaly_types, outlier_classes, files_train, files_test = mvtec_all_classes(c,
                                                                                                                   '../datasets/mvtec',
                                                                                                                   n_anom_per_cls=na,
                                                                                                                   seed=s)

            results_aexad = f'/media/rosariaspada/Dati/AE-XAD-v2/results/mvtec_all/{c}/{s}/{na}'
            results_fcdd = f'/media/rosariaspada/Dati/AE-XAD-v2/results/mvtec_all/'

            ht_aexad_res, sc_aexad, _ = load_data_aexad(results_aexad, 'aexad_htmaps_conv_deep_v2.npy', 'aexad_scores_conv_deep_v2.npy',
                                                        shape=GT_test.shape[:3])
            ht_aexad_blur = blurred_htmaps(ht_aexad_res, scale=1.0)
            ht_fabrizio = ht_aexad_res * ht_aexad_blur
            sc_fabrizio = ht_fabrizio.sum(axis=(2, 1)) / (448 * 448)
            ht_fcdd_res, sc_fcdd = load_data_fcdd(results_fcdd, c, na, s, shape=GT_test.shape[:3])

            an_gt = GT_test[:, :, :, 0][anomaly_types != 'good'] / 255.
            an_ht_fcdd = ht_fcdd_res[anomaly_types != 'good']
            an_ht_aexad = ht_aexad_blur[anomaly_types != 'good']
            an_data = X_test[anomaly_types != 'good']


            xaucs_aexad = np.empty(an_gt.shape[0])
            xaucs_fcdd = np.empty(an_gt.shape[0])
            for i in range(an_gt.shape[0]):
                xaucs_aexad[i] = average_precision(an_gt[i], an_ht_aexad[i])
                xaucs_fcdd[i] = average_precision(an_gt[i], an_ht_fcdd[i])
            if classes_names[c] != 'carpet':
                print(f'& {classes_names[c]} & {xaucs_aexad.mean():.4f} &  & {xaucs_fcdd.mean():.4f} & & {roc_auc_score(Y_test, sc_fabrizio):.4f} &  & {roc_auc_score(Y_test, sc_fcdd):.4f} \\\\')
            else:
                print('\\multirow{5}{*}{\\rot{Textures}} ',
                    f'& {classes_names[c]} & {xaucs_aexad.mean():.4f} &  & {xaucs_fcdd.mean():.4f} & {roc_auc_score(Y_test, sc_fabrizio):.4f} &  & {roc_auc_score(Y_test, sc_fcdd):.4f} \\\\')

    elif 'btad' in dataset:
        root = f'../../AE-XAD_repo/results/{dataset}/{na}'
        root_comp = f'../results/{dataset}'

        gts_aexad = np.load(os.path.join(root, 'aexad_gtmaps_norm.npy'))[:, 0]
        y = np.load(os.path.join(root, 'aexad_labels_norm.npy'))
        ht_aexad = np.load(os.path.join(root, f'aexad_htmaps_conv_deep_v2.npy'))
        sc_aexad = np.load(os.path.join(root, f'aexad_scores_conv_deep_v2.npy'))
        ht_aexad = ht_aexad.mean(axis=1)

        ht_aexad_blur = np.empty((ht_aexad.shape[0], ht_aexad.shape[1], ht_aexad.shape[2]))
        ht_aexad_blur[:, :, :, ] = blurred_htmaps(ht_aexad, scale=1.)

        gts_fcdd = np.load(os.path.join(root_comp, 'fcdd_gtmaps.npy'))[:, 0]
        ht_fcdd = np.load(os.path.join(root_comp, 'fcdd_htmaps.npy'))
        sc_fcdd = np.load(os.path.join(root_comp, 'fcdd_scores.npy'))
        gts_dev = np.load(os.path.join(root_comp, 'deviation_gtmaps.npy'))[:, 0]
        ht_dev = np.load(os.path.join(root_comp, 'deviation_htmaps.npy'))
        sc_dev = np.load(os.path.join(root_comp, 'deviation_scores.npy'))

        nat = int(y.sum())
        ht_auc_aexad_blur = np.empty((nat))
        ht_auc_fcdd = np.empty((nat))
        ht_auc_dev = np.empty((nat))
        for i in range(nat):
                ht_auc_aexad_blur[i] = average_precision(gts_aexad[y == 1][i], ht_aexad_blur[y == 1][i, :, :])
                ht_auc_fcdd[i] = average_precision(gts_fcdd[y == 1][i], ht_fcdd[y == 1][i, :, :])
                ht_auc_dev[i] = average_precision(gts_dev[y == 1][i], ht_dev[y == 1][i, :, :])

        print(f'{dataset} & {ht_auc_aexad_blur.mean()} & {ht_auc_fcdd.mean()} & {ht_auc_dev.mean()} & '
              f'{roc_auc_score(y, sc_aexad)} & {roc_auc_score(y, sc_dev)} & {roc_auc_score(y, sc_fcdd)} \\\\')