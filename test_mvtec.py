import os
import time

import PIL
import torch
import torch.utils
import torch.utils.data
from sklearn.metrics import roc_auc_score
from tools.create_dataset import mvtec
from torchvision.datasets import FakeData
#from torchvision.transforms import v2
from torchvision import transforms
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random_shape
from aexad_script import launch as launch_aexad #<----
from competitors.fcdd.run_fcdd import launch as launch_fcdd
from competitors.deviation.run_deviation import launch as launch_deviation
from competitors.bgad.run_bgad import launch as launch_bgad
from PIL import Image
from tools.evaluation_metrics import area, perimeter, iou, precision, recall, Xaucs
from tools.utils import blurred_htmaps, plots_std_bin

from codecarbon import EmissionsTracker
import json


def Xaucs(GT, hs, ids_anomalies):
    Xaucs = np.zeros(ids_anomalies.shape[0])
    for i in range(len(ids_anomalies)):
        h = hs[ids_anomalies[i]]
        Xaucs[i] = roc_auc_score(GT[ids_anomalies[i]].flatten().astype(int), h.flatten())
    return Xaucs


ds = 'mvtec'
s = 40
na = 3
only_anom = False
img_size = 224
ae_type = 'vgg_cnn'

def ds_to_numpy(ds):
    np_ds = np.zeros([ds.size]+ds.image_size)
    for i in range(ds.size):
        img, label = ds[i]
        img = img - torch.min(img)
        img = img / torch.max(img)
        img = img * 255
        np_ds[i] = np.asarray(img, dtype=np.uint8)
    return np_ds



def get(id, ds):
    img, label = ds[id]
    img = img.detach().numpy()
    img = img - np.min(img)
    img = img / np.max(img)
    return img, id, label

def plot(img_id, ds, r, c, i):
    img, id, label = get(img_id, ds)
    img = img.transpose([1, 2, 0])
    plt.subplot(r,c,i)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(f"cl {label}")
    plt.imshow(img)

preproc0 = transforms.Compose([
    transforms.PILToTensor(),
    #transforms.RandomResizedCrop(size=(32, 32), antialias=True),
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.ToDtype(torch.float32)#,  # to float32 in [0, 1]
    transforms.ToTensor()
    #transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # typically from ImageNet
])

preprocresh = transforms.Compose([
    # ------- RESIZE ---------
    transforms.ToPILImage(),
    transforms.Resize((img_size, img_size), Image.NEAREST),
    # ------------------------
    #transforms.ToTensor()
])

preproc1 = transforms.Compose([
    #transforms.PILToTensor(),
    #transforms.RandomResizedCrop(size=(32, 32), antialias=True),
#   # ------- RESIZE ---------
    transforms.ToPILImage(),
    transforms.Resize((img_size, img_size), Image.NEAREST),
    # -------------------------
    ##transforms.GaussianBlur(127),
    ##transforms.GaussianBlur(127),
    ##transforms.GaussianBlur(127),
    ##transforms.GaussianBlur(127),
    ##transforms.GaussianBlur(127),
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.ToDtype(torch.float32)#, # to float32 in [0, 1]
    #transforms.ToTensor()
    #transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # typically from ImageNet
])


dataset = ds

if not os.path.exists(f'PLOT/{dataset}'):
    os.makedirs(f'PLOT/{dataset}')

for c in range(1, 15):
    if not os.path.exists(f'PLOT/{dataset}/{c}'):
        os.makedirs(f'PLOT/{dataset}/{c}')


    X_train, Y_train, X_test, Y_test, GT_train, GT_test = mvtec(c, 'datasets/mvtec', na, seed=s)

    if only_anom:
        X_train = X_train[Y_train == 1]
        GT_train = GT_train[Y_train == 1]
        Y_train = Y_train[Y_train == 1]
        X_test = X_test[Y_test == 1]
        GT_test = GT_test[Y_test == 1]
        Y_test = Y_test[Y_test == 1]

    print(X_train.max(), GT_train.max(), X_test.max(), GT_test.max())

    X_train_resh = np.zeros((X_train.shape[0], 3, img_size, img_size))
    GT_train_resh = np.zeros((GT_train.shape[0], 3, img_size, img_size))
    X_test_resh = np.zeros((X_test.shape[0], 3, img_size, img_size))
    GT_test_resh = np.zeros((GT_test.shape[0], 3, img_size, img_size))
    for i in range(X_train.shape[0]):
        X_train_resh[i, :, :, :] = np.array(preproc1(X_train[i, :, :, :])).swapaxes(2, 1).swapaxes(1, 0)
        GT_train_resh[i, :, :, :] = np.array(preproc1(GT_train[i, :, :, :])).swapaxes(2, 1).swapaxes(1, 0)
    for i in range(X_test.shape[0]):
        X_test_resh[i, :, :, :] = np.array(preproc1(X_test[i, :, :, :])).swapaxes(2, 1).swapaxes(1, 0)
        GT_test_resh[i, :, :, :] = np.array(preproc1(GT_test[i, :, :, :])).swapaxes(2, 1).swapaxes(1, 0)

    X_train = X_train_resh
    GT_train = GT_train_resh
    X_test = X_test_resh
    GT_test = GT_test_resh


    print(GT_test.shape, X_train.max(), GT_train.max(), X_test.max(), GT_test.max())


    data_path = os.path.join('datasets', ds)

    np.save(os.path.join(data_path, 'X_train.npy'), X_train)
    np.save(os.path.join(data_path, 'Y_train.npy'), Y_train)
    np.save(os.path.join(data_path, 'GT_train.npy'), GT_train)

    np.save(os.path.join(data_path, 'X_test.npy'), X_test)
    np.save(os.path.join(data_path, 'Y_test.npy'), Y_test)
    np.save(os.path.join(data_path, 'GT_test.npy'), GT_test)


    #def f(x):
    #    return torch.where(x >= 0.5, 0.0, 1.0)


    #def f_np(x):
    #    return np.where(x >= 0.5, 0.0, 1.0)

    def f(x):
        return torch.full_like(x, fill_value=2.)

    def f_np(x):
        return np.full_like(x, fill_value=2.)


    emissions = []
    times = []
    methods = ['FCDD', 'AE-XAD', 'BGAD', 'AE']

    del X_train, Y_train, GT_train

    ret_path = f'results/{ds}_final_f2/{c}/{s}'

    if not os.path.exists(ret_path):
        os.makedirs(ret_path)

    tracker = EmissionsTracker(tracking_mode='process', log_level='critical')
    tracker.start_task('fcdd')
    t = time.time()
    heatmaps_fcdd, scores_fcdd, gtmaps_fcdd, labels_fcdd, time_fcdd = launch_fcdd(data_path, epochs=1,
                                                                                  batch_size=16)  # 200
    times.append(time.time() - t)
    emissions.append(tracker.stop_task('fcdd').emissions)

    tracker.start_task('aexad')
    t = time.time()
    heatmaps_aexad, scores_aexad, _, _, time_aexad = launch_aexad(data_path, 200, 16, 64, None,
                                                                                        None, f, ae_type,
                                                                                        save_intermediate=False,
                                                                                        save_path=ret_path,
                                                                                        use_cuda=True,
                                                                                        dataset='tf_ds',
                                                                                        loss='aexad')  # 400
    times.append(time.time() - t)
    emissions.append(tracker.stop_task('aexad').emissions)
    print(roc_auc_score(Y_test, scores_aexad))

    tracker.start_task('bgad')
    t = time.time()
    # heatmaps_devnet, scores_devnet, gtmaps_devnet, labels_devnet, tot_time_devnet = launch_deviation(dataset_root=data_path, epochs=50, img_size=224)
    heatmaps_bgad, scores_bgad, _, _ = launch_bgad(data_path)
    times.append(time.time() - t)
    emissions.append(tracker.stop_task('bgad').emissions)

    tracker.start_task('ae')
    t = time.time()
    heatmaps_ae, scores_ae, _, _, time_ae = launch_aexad(data_path, 1, 16, 64, None, None, f,
                                                                         ae_type,
                                                                         save_intermediate=False,
                                                                         save_path=ret_path,
                                                                         use_cuda=True,
                                                                         dataset='tf_ds', loss='mse')  # 400
    times.append(time.time() - t)
    emissions.append(tracker.stop_task('ae').emissions)

    d = {'methods': methods, 'times': times, 'emissions': emissions}

    tracker.stop()

    # ------ PREPARAZIONE HEATMAP
    imgs = X_test / 255.
    np.save(os.path.join(ret_path, f'heatmaps_aexad_{ae_type}.npy'), heatmaps_aexad)
    heatmaps_aexad_norm = heatmaps_aexad / (f_np(imgs) - imgs) ** 2
    heatmaps_aexad_norm = heatmaps_aexad_norm.mean(axis=1)
    heatmaps_aexad_norm = np.where(heatmaps_aexad_norm>1., 1., heatmaps_aexad_norm)
    heatmaps_aexad = heatmaps_aexad.mean(axis=1)
    heatmaps_aexad_norm_blur = blurred_htmaps(heatmaps_aexad_norm, thres=0.3, scale=0.5)
    heatmaps_ae = heatmaps_ae.mean(axis=1)
    np.save(os.path.join(ret_path, f'heatmaps_ae_{ae_type}.npy'), heatmaps_ae)
    np.save(os.path.join(ret_path, 'heatmmaps_fcdd.npy'), heatmaps_fcdd)
    np.save(os.path.join(ret_path, 'heatmaps_bgad.npy'), heatmaps_bgad)

    np.save(os.path.join(ret_path, 'scores_aexad.npy'), scores_aexad)
    np.save(os.path.join(ret_path, 'scores_fcdd.npy'), scores_fcdd)
    np.save(os.path.join(ret_path, 'scores_bgad.npy'), scores_bgad)
    np.save(os.path.join(ret_path, 'scores_ae.npy'), scores_ae)

    ids_anomalies = np.where(Y_test == 1)[0]


    def plots(i, ids_anomalies, x, gt, hs1, hs2, hs3, xaucs1, xaucs2, xaucs3, num_std=None, c=None):
        id = ids_anomalies[i]
        plt.subplot(1, 5, 1)
        plt.imshow(x[id], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('IMG')
        plt.subplot(1, 5, 2)
        plt.imshow(gt[id], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('GT')
        plt.subplot(1, 5, 3)
        plt.imshow(hs1[id], cmap='jet')
        plt.xticks([])
        plt.yticks([])
        # plt.title(f'AE-XAD - SCORE: {scores_aexad[id]:.3f} - XAUC: {xaucs1[i]:.3f}')
        plt.title(f'AE-XAD')
        plt.xlabel(f'XAUC: {xaucs1[i]:.3f}')
        plt.subplot(1, 5, 4)
        plt.imshow(hs2[id], cmap='jet')
        plt.xticks([])
        plt.yticks([])
        # plt.title(f'FCDD - SCORE: {scores_fcdd[id]:.3f} - XAUC: {xaucs2[i]:.3f}')
        plt.title(f'FCDD')
        plt.xlabel(f'XAUC: {xaucs2[i]:.3f}')
        plt.subplot(1, 5, 5)
        plt.imshow(hs3[id], cmap='jet')
        plt.xticks([])
        plt.yticks([])
        # plt.title(f'FCDD - SCORE: {scores_fcdd[id]:.3f} - XAUC: {xaucs2[i]:.3f}')
        plt.title(f'BGAD')
        plt.xlabel(f'XAUC: {xaucs3[i]:.3f}')
        plt.tight_layout()
        pr = perimeter(gt[id])
        ar = area(gt[id])
        plt.suptitle(f"ID: {id} - P/A: {pr / ar:.3f} - P^2/A: {(pr ** 2) / ar:.3f}", fontsize=10, fontweight='bold')
        if num_std == None:
            if c == None:
                name = f'PLOT/{dataset}/fig_{i}.jpg'
            else:
                name = f'PLOT/{dataset}/{c}/fig_{i}.jpg'
        else:
            if c == None:
                name = f'PLOT/{dataset}/fig_{i}_{num_std}.jpg'
            else:
                name = f'PLOT/{dataset}/{c}/fig_{i}_{num_std}.jpg'
        plt.savefig(name)


    GT_test = GT_test[:, 0] / 255.
    heatmaps_fcdd = heatmaps_fcdd.mean(axis=1)
    gtmaps_fcdd = gtmaps_fcdd[:, 0]

    xaucs_aexad = np.nan_to_num(Xaucs(GT_test, heatmaps_aexad_norm_blur, ids_anomalies))
    xaucs_fcdd = np.nan_to_num(Xaucs(gtmaps_fcdd, heatmaps_fcdd, ids_anomalies))
    # xaucs_devnet = Xaucs(np.array(gtmaps_devnet)[:, 0]//255, np.array(heatmaps_devnet))
    xaucs_bgad = np.nan_to_num(Xaucs(GT_test, heatmaps_bgad, ids_anomalies))
    xaucs_ae = np.nan_to_num(Xaucs(GT_test, heatmaps_ae, ids_anomalies))

    d['aucs'] = {'aexad': list(xaucs_aexad),
                 'fcdd': list(xaucs_fcdd),
                 'bgad': list(xaucs_bgad),
                 'ae': list(xaucs_ae)}


    #def f_np(x):
    #    return np.where(x >= 0.5, 0.0, 1.0)


    for i in range(Y_test[Y_test==1].shape[0]):
        plots(i, ids_anomalies, imgs.transpose((0, 2, 3, 1)), GT_test,
              heatmaps_aexad_norm_blur, heatmaps_fcdd, heatmaps_bgad,
              xaucs_aexad, xaucs_fcdd, xaucs_bgad, c=c)

    mean_aexad = heatmaps_aexad_norm_blur.mean()
    mean_fcdd = heatmaps_fcdd.mean()
    mean_bgad = heatmaps_bgad.mean()
    mean_ae = heatmaps_ae.mean()
    std_aexad = heatmaps_aexad_norm_blur.std()
    std_fcdd = heatmaps_fcdd.std()
    std_bgad = heatmaps_bgad.std()
    std_ae = heatmaps_ae.std()

    d_acc = {}

    for num_std in np.arange(1., 4., 0.5):
        print('NUM STD: ', num_std)
        ht_aexad_bin = np.where((heatmaps_aexad_norm_blur - mean_aexad) / std_aexad >= num_std, 1., 0.)
        ht_fcdd_bin = np.where((heatmaps_fcdd - mean_fcdd) / std_fcdd >= num_std, 1., 0.)
        ht_bgad_bin = np.where((heatmaps_bgad - mean_bgad) / std_bgad >= num_std, 1., 0.)
        ht_ae_bin = np.where((heatmaps_ae - mean_ae) / std_ae >= num_std, 1., 0.)

        iou_aexad = np.empty(ht_aexad_bin[Y_test == 1].shape[0], dtype=float)
        iou_fcdd = np.empty(ht_aexad_bin[Y_test == 1].shape[0], dtype=float)
        iou_bgad = np.empty(ht_aexad_bin[Y_test == 1].shape[0], dtype=float)
        iou_ae = np.empty(ht_aexad_bin[Y_test == 1].shape[0], dtype=float)
        pre_aexad = np.empty(ht_aexad_bin[Y_test == 1].shape[0], dtype=float)
        pre_fcdd = np.empty(ht_aexad_bin[Y_test == 1].shape[0], dtype=float)
        pre_bgad = np.empty(ht_aexad_bin[Y_test == 1].shape[0], dtype=float)
        pre_ae = np.empty(ht_aexad_bin[Y_test == 1].shape[0], dtype=float)
        rec_aexad = np.empty(ht_aexad_bin[Y_test == 1].shape[0], dtype=float)
        rec_fcdd = np.empty(ht_aexad_bin[Y_test == 1].shape[0], dtype=float)
        rec_bgad = np.empty(ht_aexad_bin[Y_test == 1].shape[0], dtype=float)
        rec_ae = np.empty(ht_aexad_bin[Y_test == 1].shape[0], dtype=float)
        for i in range(ht_aexad_bin[Y_test == 1].shape[0]):
            iou_aexad[i] = iou(GT_test[Y_test == 1][i], ht_aexad_bin[Y_test == 1][i])
            pre_aexad[i] = precision(GT_test[Y_test == 1][i], ht_aexad_bin[Y_test == 1][i])
            rec_aexad[i] = recall(GT_test[Y_test == 1][i], ht_aexad_bin[Y_test == 1][i])
            iou_fcdd[i] = iou(gtmaps_fcdd[Y_test == 1][i], ht_fcdd_bin[Y_test == 1][i])
            pre_fcdd[i] = precision(gtmaps_fcdd[Y_test == 1][i], ht_fcdd_bin[Y_test == 1][i])
            rec_fcdd[i] = recall(gtmaps_fcdd[Y_test == 1][i], ht_fcdd_bin[Y_test == 1][i])
            iou_bgad[i] = iou(GT_test[Y_test == 1][i], ht_bgad_bin[Y_test == 1][i])
            pre_bgad[i] = precision(GT_test[Y_test == 1][i], ht_bgad_bin[Y_test == 1][i])
            rec_bgad[i] = recall(GT_test[Y_test == 1][i], ht_bgad_bin[Y_test == 1][i])
            iou_ae[i] = iou(GT_test[Y_test == 1][i], ht_ae_bin[Y_test == 1][i])
            pre_ae[i] = precision(GT_test[Y_test == 1][i], ht_ae_bin[Y_test == 1][i])
            rec_ae[i] = recall(GT_test[Y_test == 1][i], ht_ae_bin[Y_test == 1][i])

        d_acc[num_std] = {'aexad': {'iou': list(iou_aexad), 'pre': list(pre_aexad), 'rec': list(rec_aexad)},
                          'fcdd': {'iou': list(iou_fcdd), 'pre': list(pre_fcdd), 'rec': list(rec_fcdd)},
                          'bgad': {'iou': list(iou_bgad), 'pre': list(pre_bgad), 'rec': list(rec_bgad)},
                          'ae': {'iou': list(iou_ae), 'pre': list(pre_ae), 'rec': list(rec_ae)}}

        # for i in range(15):
        #    plots(i, ids_anomalies, imgs.transpose((0, 2, 3, 1)), GT_test,
        #          ht_aexad_bin, ht_fcdd_bin, ht_bgad_bin,
        #          xaucs_aexad, xaucs_fcdd, xaucs_bgad, num_std=num_std)

    imgs_aexad = []
    imgs_fcdd = []
    imgs_bgad = []
    imgs_ae = []
    for num_std in np.arange(1., 4., 0.5):
        imgs_aexad.append(np.where((heatmaps_aexad_norm_blur - mean_aexad) / std_aexad >= num_std, 1., 0.))
        imgs_fcdd.append(np.where((heatmaps_fcdd - mean_fcdd) / std_fcdd >= num_std, 1., 0.))
        imgs_bgad.append(np.where((heatmaps_bgad - mean_bgad) / std_bgad >= num_std, 1., 0.))
        imgs_ae.append(np.where((heatmaps_ae - mean_ae) / std_ae >= num_std, 1., 0.))

    for i in range(Y_test[Y_test == 1].shape[0]):
        plots_std_bin(i, ids_anomalies, imgs.transpose((0, 2, 3, 1)), GT_test,
                      imgs_aexad, imgs_ae, imgs_fcdd, imgs_bgad, np.arange(1., 4., 0.5), dataset, c)

    d['bin_stats'] = d_acc

    with open(os.path.join(ret_path, 'stats.txt'), 'w') as file:
        file.write(json.dumps(d))

    metric = 'aucs'
    methods = ['aexad', 'fcdd', 'bgad']
    print(methods)
    for method in methods:
        print(np.array(d[metric][method]).mean())

    for num_std in np.arange(1., 4., 0.5):
        bs = d['bin_stats'][num_std]
        aexad = bs['aexad']
        fcdd = bs['fcdd']
        bgad = bs['bgad']
        print(
            f"{num_std} & IOU: {np.array(fcdd['iou']).mean()} & {np.array(aexad['iou']).mean()} & {np.array(bgad['iou']).mean()}")
        print(
            f"{num_std} & PRE: {np.array(fcdd['pre']).mean()} & {np.array(aexad['pre']).mean()} & {np.array(bgad['pre']).mean()}")
        print(
            f"{num_std} & REC: {np.array(fcdd['rec']).mean()} & {np.array(aexad['rec']).mean()} & {np.array(bgad['rec']).mean()}")
        print('-------------------------------------')
