
import os

import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

from tools.create_dataset import mvtec_only_one, mvtec, mvtec_aug, mvtec_all_classes
from tools.evaluation_metrics import Xauc, precision, recall, iou
from tools.load_data import load_data_aexad, load_data_fcdd, load_data_devnet, load_data_all
from tools.utils import blurred_htmaps, plot_ht_auc

def f(x):
    return np.where(x>0.5, 0., 1.)

import numpy as np

if __name__ == '__main__':
    labels = (
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper'
    )

    dataset = 'btad_01'
    ds = ''
    root = f'results/{dataset}'
    model = 'conv_deep_v2'
    na = 10
    s = 40
    classes = list(range(1,2))#15

    transform_aexad = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((448, 448), Image.NEAREST),
        transforms.PILToTensor(),
    ])

    transform_fcdd = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224), Image.NEAREST),
        transforms.PILToTensor(),
    ])

    print('& $\\aexad$ & $\\fcdd$ & $\\aexad$ & $\\fcdd$ & $\\aexad$ & $\\fcdd$ & $\\aexad$ & $\\fcdd$ & $\\aexad$ & $\\aexad$ (sc. Fab.) & $\\fcdd$ \\\\')
    for c in classes:
        if 'mvtec' in dataset:
            ds = labels[c]
            if dataset == 'mvtec':
                X_train, Y_train, X_test, Y_test, GT_train, GT_test = mvtec(c, 'datasets/mvtec',
                                                                                     n_anom_per_cls=na,
                                                                                     seed=s)
                root = f'results/f_1/{dataset}/{c}/{s}/{na}'
                root_comp = f'/media/rosariaspada/Dati/AE-XAD-v2/results/{dataset}/'
            if dataset == 'mvtec_all':
                _, _, X_test, Y_test, _, GT_test, anomaly_types, outlier_classes, files_train, files_test = mvtec_all_classes(
                    c,
                    'datasets/mvtec',
                    n_anom_per_cls=na,
                    seed=s)

                #root = f'results/only_anom/mvtec_all/{c}/{s}/{na}'
                #root_comp = f'/media/rosariaspada/Dati/AE-XAD-v2/results/only_anom/mvtec_all/'
                root = f'results/mvtec_all/{c}/{s}/{na}'
                root_comp = f'/media/rosariaspada/Dati/AE-XAD-v2/results/mvtec_all/'

            hts_path = f'aexad_htmaps_{model}.npy'
            scores_path = f'aexad_scores_{model}.npy'
            model_path = f'aexad_model_{model}.pt'

            ht_fcdd_res, sc_fcdd = load_data_fcdd(root_comp, c, na, s, shape=GT_test.shape[:3])
            ht_aexad, _, _ = load_data_aexad(root, hts_path, scores_path,
                                             shape=(X_test.shape[0], 3, X_test.shape[1], X_test.shape[2]),
                                             model_file=model_path, lm=False)

        if 'btad' in dataset:
            X_test = np.load(open(os.path.join(f'../AE-XAD_repo/data/{dataset}', 'X_test.npy'), 'rb'))
            Y_test = np.load(open(os.path.join(f'../AE-XAD_repo/data/{dataset}', 'Y_test.npy'), 'rb'))
            GT_test = np.load(open(os.path.join(f'../AE-XAD_repo/data/{dataset}', 'GT_test.npy'), 'rb'))

            root = f'../AE-XAD_repo/results/{dataset}/{na}'
            model = 'conv_deep_v2'
            gts = np.load(os.path.join(root, 'aexad_gtmaps_norm.npy'))[:, 0]
            y = np.load(os.path.join(root, 'aexad_labels_norm.npy'))
            ht_aexad = np.load(os.path.join(root, f'aexad_htmaps_{model}.npy')).sum(axis=1)
            sc_aexad = np.load(os.path.join(root, f'aexad_scores_{model}.npy'))

            root_comp = f'results/{dataset}'
            ht_fcdd_res = np.load(os.path.join(root_comp, f'fcdd_htmaps.npy'))[:,0]
            sc_fcdd = np.load(os.path.join(root_comp, f'fcdd_scores.npy'))


        X_test_aexad = np.empty((X_test.shape[0], 3, 448, 448))
        GT_test_aexad = np.empty((X_test.shape[0], 3, 448, 448))
        GT_test_fcdd = np.empty((X_test.shape[0], 3, 224, 224))
        for i in range(X_test.shape[0]):
            X_test_aexad[i] = transform_aexad(X_test[i]).numpy()
            GT_test_aexad[i] = transform_aexad(GT_test[i]).numpy()
            GT_test_fcdd[i] = transform_fcdd(GT_test[i]).numpy()


        X_test_aexad = X_test_aexad.swapaxes(1,2).swapaxes(2,3) / 255.
        GT_test_aexad = GT_test_aexad[:, 0] // 255
        GT_test_fcdd = GT_test_fcdd[:, 0] // 255


        ht_norm = ht_aexad / ((f(X_test_aexad) - X_test_aexad)**2).sum(axis=-1)#np.abs(f(X_test_aexad) - X_test_aexad).sum(axis=-1)
        ht_norm = np.where(ht_norm >= 1., 1., ht_norm)             # Eventuali errori di approssimazione
        ht_aexad = ht_aexad / 3.
        ht_aexad_blur = blurred_htmaps(ht_aexad, scale=0.5)
        ht_aexad_blur_norm = blurred_htmaps(ht_norm, scale=0.5)
        ht_fabrizio = ht_aexad * ht_aexad_blur
        sc_fabrizio = ht_fabrizio.sum(axis=(2, 1)) / np.prod(GT_test_aexad.shape[1:])
        sc_aexad = ht_norm.sum(axis=(2, 1))

        mean_aexad = ht_aexad_blur_norm.mean()
        mean_fcdd = ht_fcdd_res.mean()
        std_aexad = ht_aexad_blur_norm.std()
        std_fcdd = ht_fcdd_res.std()
        num_std = 3
        ht_aexad_bin = np.where((ht_aexad_blur_norm - mean_aexad) / std_aexad >= num_std, 1., 0.)
        ht_fcdd_bin = np.where((ht_fcdd_res - mean_fcdd) / std_fcdd >= num_std, 1., 0.)

        ht_auc_aexad = np.empty(GT_test_aexad[Y_test == 1].shape[0])
        ht_auc_aexad_blur = np.empty(GT_test_aexad[Y_test == 1].shape[0])
        ht_auc_aexad_norm = np.empty(GT_test_aexad[Y_test == 1].shape[0])
        ht_auc_aexad_norm_blur = np.empty(GT_test_aexad[Y_test == 1].shape[0])
        ht_auc_fcdd = np.empty(GT_test_aexad[Y_test == 1].shape[0])

        ht_iou_aexad = np.empty(GT_test_aexad[Y_test == 1].shape[0])
        ht_pre_aexad = np.empty(GT_test_aexad[Y_test == 1].shape[0])
        ht_rec_aexad = np.empty(GT_test_aexad[Y_test == 1].shape[0])
        ht_iou_fcdd = np.empty(GT_test_aexad[Y_test == 1].shape[0])
        ht_pre_fcdd = np.empty(GT_test_aexad[Y_test == 1].shape[0])
        ht_rec_fcdd = np.empty(GT_test_aexad[Y_test == 1].shape[0])

        for i in range(GT_test_aexad[Y_test == 1].shape[0]):
            ht_auc_aexad[i] = Xauc(GT_test_aexad[Y_test == 1][i], ht_aexad[Y_test == 1][i])
            ht_auc_aexad_blur[i] = Xauc(GT_test_aexad[Y_test == 1][i], ht_aexad_blur[Y_test == 1][i])
            ht_auc_aexad_norm[i] = Xauc(GT_test_aexad[Y_test == 1][i], ht_norm[Y_test == 1][i]) #_fabrizio
            ht_auc_aexad_norm_blur[i] = Xauc(GT_test_aexad[Y_test == 1][i], ht_aexad_blur_norm[Y_test == 1][i])
            ht_auc_fcdd[i] = Xauc(GT_test_fcdd[Y_test == 1][i], ht_fcdd_res[Y_test == 1][i])

            ht_iou_aexad[i] = iou(GT_test_aexad[Y_test==1][i], ht_aexad_bin[Y_test==1][i])
            ht_pre_aexad[i] = precision(GT_test_aexad[Y_test==1][i], ht_aexad_bin[Y_test==1][i])
            ht_rec_aexad[i] = recall(GT_test_aexad[Y_test==1][i], ht_aexad_bin[Y_test==1][i])
            ht_iou_fcdd[i] = iou(GT_test_fcdd[Y_test==1][i], ht_fcdd_bin[Y_test==1][i])
            ht_pre_fcdd[i] = precision(GT_test_fcdd[Y_test==1][i], ht_fcdd_bin[Y_test==1][i])
            ht_rec_fcdd[i] = recall(GT_test_fcdd[Y_test==1][i], ht_fcdd_bin[Y_test==1][i])

        ht_iou_aexad = np.nan_to_num(ht_iou_aexad, nan=0.)
        ht_pre_aexad = np.nan_to_num(ht_pre_aexad, nan=0.)
        ht_rec_aexad = np.nan_to_num(ht_rec_aexad, nan=0.)
        ht_iou_fcdd = np.nan_to_num(ht_iou_fcdd, nan=0.)
        ht_pre_fcdd = np.nan_to_num(ht_pre_fcdd, nan=0.)
        ht_rec_fcdd = np.nan_to_num(ht_rec_fcdd, nan=0.)

        #print(f'------------------- {labels[c]} --------------------')
        # print('Explanation')
        print(ds, end=' & ')
        # print(np.round(ht_auc_aexad.mean(), 4), end=' & ')
        # print(np.round(ht_auc_aexad_norm.mean(), 4), end=' & ')
        # print(np.round(ht_auc_aexad_blur.mean(), 4), end=' & ')
        # print(np.round(ht_auc_aexad_norm_blur.mean(), 4), end=' \\\\\n ')
        #print('& AUC & IOU & PRE & REC')
        #print(f'$\\aexad$ & {ht_auc_aexad_norm_blur.mean():.4f} & {ht_iou_aexad.mean():.4f} & {ht_pre_aexad.mean():.4f} & {ht_rec_aexad.mean():.4f} \\\\')
        #print(f'$\\fcdd$ & {ht_auc_fcdd.mean():.4f} & {ht_iou_fcdd.mean():.4f} & {ht_pre_fcdd.mean():.4f} & {ht_rec_fcdd.mean():.4f}')

        #print('Detection')
        #print(ds, end=' & ')
        #print(np.round(roc_auc_score(Y_test, sc_aexad), 4), end=' & ')
        #print(np.round(roc_auc_score(Y_test, sc_fabrizio), 4), end=' \\\ ')
        #print(np.round(roc_auc_score(Y_test, sc_fcdd), 4), end=' \\\\\n ')


        # Stampa tabella
        print(f' & {ht_auc_aexad_norm_blur.mean():.4f} & {ht_auc_fcdd.mean():.4f}', end=' & ')
        print(f'{ht_iou_aexad.mean():.4f} & {ht_iou_fcdd.mean():.4f}', end=' & ')
        print(f'{ht_pre_aexad.mean():.4f} & {ht_pre_fcdd.mean():.4f}', end=' & ')
        print(f'{ht_rec_aexad.mean():.4f} & {ht_rec_fcdd.mean():.4f}', end=' & ')
        print(np.round(roc_auc_score(Y_test, sc_aexad), 4), end=' & ')
        print(np.round(roc_auc_score(Y_test, sc_fabrizio), 4), end=' & ')
        print(np.round(roc_auc_score(Y_test, sc_fcdd), 4), end=' \\\\\n ')
