import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import warn

from tools.create_dataset import square, square_diff
from scipy.ndimage import gaussian_filter

from tools.evaluation_metrics import Xauc, perimeter, area


def weighted_htmaps(htmaps, n_pixels=1):
    w_htmaps = np.empty_like(htmaps)
    for k in range(len(htmaps)):
        ht = htmaps[k]
        for i in range(ht.shape[0]):
            for j in range(ht.shape[1]):
                idx_i = np.arange(i - n_pixels, i + n_pixels + 1)
                idx_i = idx_i[np.where((idx_i >= 0) & (idx_i < ht.shape[-2]))[0]]
                idx_j = np.arange(j - n_pixels, j + n_pixels + 1)
                idx_j = idx_j[np.where((idx_j >= 0) & (idx_j < ht.shape[-2]))[0]]
                s = ht[idx_i][:, idx_j].sum() - ht[i, j]
                w_htmaps[k, i, j] = ht[i, j] * (s / (len(idx_i) * len(idx_j) - 1))
    return w_htmaps


def plot_image(image, cmap='gray'):
    '''
    Method to plot an image
    :param image: (C x H x W) channel-first-format image
    :return:
    '''
    if image.shape[0] == 1:
        image = image[0, :, :]
    else:
        image = np.swapaxes(image, 0, 1)
        image = np.swapaxes(image, 1, 2)

    plt.imshow(image, cmap=cmap)
    plt.show()


def plot_heatmap(image, colorbar=True):
    '''
    Method to plot a heatmap
    :param image: (1 x H x W) channel-first-format image
    :return:
    '''
    image = image[0, :, :]

    plt.imshow(image, cmap='jet')#, vmin=0.0, vmax=1.0)
    if colorbar:
        plt.colorbar()
    plt.show()

def plot_results(path, method):
    infos = (os.path.normpath(path)).split(os.sep)
    dataset = infos[-3]
    c = int(infos[-2])
    seed = int(infos[-1])

    Y_test = np.load(open(os.path.join(path, 'labels.npy'), 'rb'))
    #perc_anom_test = np.sum(Y_test) / len(Y_test)

    _, _, X_test, Y_test, _, GT_test = square_diff(c, perc_anom_train=0.02, perc_anom_test=0.10, size=5,
               intensity=0.25, DATASET=dataset, seed=seed)

    htmaps_dev = np.load(open(os.path.join(path, 'deviation_htmaps.npy'), 'rb'))
    scores_dev = np.load(open(os.path.join(path, 'deviation_scores.npy'), 'rb'))
    htmaps_fcdd = np.load(open(os.path.join(path, 'fcdd_htmaps.npy'), 'rb'))
    scores_fcdd = np.load(open(os.path.join(path, 'fcdd_scores.npy'), 'rb'))
    htmaps_ae = np.load(open(os.path.join(path, 'ae_htmaps.npy'), 'rb'))
    scores_ae = np.load(open(os.path.join(path, 'ae_scores.npy'), 'rb'))
    htmaps_aexad_conv = np.load(open(os.path.join(path, 'aexad_htmaps.npy'), 'rb'))
    scores_aexad_conv = np.load(open(os.path.join(path, 'aexad_scores.npy'), 'rb'))

    if method == 'deviation':
        idx = np.argsort(scores_dev[Y_test==0])[::-1]
    elif method == 'fcdd':
        idx = np.argsort(scores_fcdd[Y_test==0])[::-1]
    elif method == 'ae':
       idx = np.argsort(scores_ae[Y_test==0])[::-1]
    elif method == 'aexad_conv':
        idx = np.argsort(scores_aexad_conv[Y_test==0])[::-1]

    plt.title(f'{method}')
    print(idx[:5])

    for i in range(5):
        plt.subplot(5, 5, (i*5)+1)
        #print(htmaps_aexad.shape)
        plot_image(X_test[Y_test==0][idx[i]])
        plt.subplot(5, 5, (i*5)+2)
        plot_heatmap(htmaps_dev[Y_test==0][idx[i]].reshape(1, X_test.shape[-2], X_test.shape[-1]))
        plt.subplot(5, 5, (i*5)+3)
        plot_heatmap(htmaps_fcdd[Y_test==0][idx[i]])
        plt.subplot(5, 5, (i*5)+4)
        plot_heatmap(htmaps_ae[Y_test==0][idx[i]])
        plt.subplot(5, 5, (i*5) + 5)
        plot_heatmap(htmaps_aexad_conv[Y_test == 0][idx[i]])

    plt.show()

def plot_results_anom(path, method):
    infos = (os.path.normpath(path)).split(os.sep)
    dataset = infos[-3]
    c = int(infos[-2])
    seed = int(infos[-1])

    Y_test_file = np.load(open(os.path.join(path, 'labels.npy'), 'rb'))
    perc_anom_test = np.sum(Y_test_file) / len(Y_test_file)
    print(perc_anom_test)

    _, _, X_test, Y_test, _, GT_test = square_diff(c, perc_anom_train=0.02, perc_anom_test=0.10, size=5,
               intensity=0.25, DATASET=dataset, seed=seed)

    print('TEST: ', X_test[0,0,0,0])

    htmaps_dev = np.load(open(os.path.join(path, 'deviation_htmaps.npy'), 'rb'))
    scores_dev = np.load(open(os.path.join(path, 'deviation_scores.npy'), 'rb'))
    htmaps_fcdd = np.load(open(os.path.join(path, 'fcdd_htmaps.npy'), 'rb'))
    scores_fcdd = np.load(open(os.path.join(path, 'fcdd_scores.npy'), 'rb'))
    htmaps_ae = np.load(open(os.path.join(path, 'ae_htmaps.npy'), 'rb'))
    scores_ae = np.load(open(os.path.join(path, 'ae_scores.npy'), 'rb'))
    #htmaps_aexad_lenet = np.load(open(os.path.join(path.replace('mnist', 'conv/mnist'), 'aexad_htmaps_lenet.npy'), 'rb'))
    #scores_aexad_lenet = np.load(open(os.path.join(path.replace('mnist', 'conv/mnist'), 'aexad_scores_lenet.npy'), 'rb'))
    htmaps_aexad_conv = np.load(open(os.path.join(path, 'aexad_htmaps_conv.npy'), 'rb'))
    scores_aexad_conv = np.load(open(os.path.join(path, 'aexad_scores_conv.npy'), 'rb'))

    if method == 'deviation':
        idx = np.argsort(scores_dev[Y_test==1])
    elif method == 'fcdd':
        idx = np.argsort(scores_fcdd[Y_test==1])
    elif method == 'ae':
        idx = np.argsort(scores_ae[Y_test==1])
    elif method == 'aexad_conv':
        idx = np.argsort(scores_aexad_conv[Y_test == 1])


    plt.title(f'{method}')
    print(idx[:5])

    for i in range(5):
        plt.subplot(5, 6, (i*6)+1)
        plot_image(X_test[Y_test==1][idx[i]])
        plt.subplot(5, 6, (i*6)+2)
        plt.title('DEV')
        plot_heatmap(htmaps_dev[Y_test==1][idx[i]].reshape(1, X_test.shape[-2], X_test.shape[-1]))
        plt.subplot(5, 6, (i*6)+3)
        plt.title('FCDD')
        plot_heatmap(htmaps_fcdd[Y_test==1][idx[i]])
        plt.subplot(5, 6, (i*6)+4)
        plt.title('AE')
        plot_heatmap(htmaps_ae[Y_test==1][idx[i]])
        plt.subplot(5, 6, (i*6)+5)
        plt.title('AEXAD')
        plot_heatmap(htmaps_aexad_conv[Y_test==1][idx[i]])
        plt.subplot(5, 6, (i*6)+6)
        #intensity = GT_test[Y_test==1][idx[i]]*X_test[Y_test==1][idx[i]]
        #plt.title(np.unique(intensity).sum())
        plot_heatmap(GT_test[Y_test==1][idx[i]])

    plt.show()

def plot_results_anom_top(path, method):
    infos = (os.path.normpath(path)).split(os.sep)
    dataset = infos[-3]
    c = int(infos[-2])
    seed = int(infos[-1])

    Y_test_file = np.load(open(os.path.join(path, 'labels.npy'), 'rb'))
    perc_anom_test = np.sum(Y_test_file) / len(Y_test_file)
    print(perc_anom_test)

    _, _, X_test, Y_test, _, GT_test = square_diff(c, perc_anom_train=0.02, perc_anom_test=0.10, size=5,
               intensity=0.25, DATASET=dataset, seed=seed)

    htmaps_dev = np.load(open(os.path.join(path, 'deviation_htmaps.npy'), 'rb'))
    scores_dev = np.load(open(os.path.join(path, 'deviation_scores.npy'), 'rb'))
    htmaps_fcdd = np.load(open(os.path.join(path, 'fcdd_htmaps.npy'), 'rb'))
    scores_fcdd = np.load(open(os.path.join(path, 'fcdd_scores.npy'), 'rb'))
    htmaps_ae = np.load(open(os.path.join(path, 'ae_htmaps.npy'), 'rb'))
    scores_ae = np.load(open(os.path.join(path, 'ae_scores.npy'), 'rb'))
    htmaps_aexad_conv = np.load(open(os.path.join(path, 'aexad_htmaps_conv.npy'), 'rb'))
    scores_aexad_conv = np.load(open(os.path.join(path, 'aexad_scores_conv.npy'), 'rb'))

    if method == 'deviation':
        idx = np.argsort(scores_dev[Y_test==1])[::-1]
    elif method == 'fcdd':
        idx = np.argsort(scores_fcdd[Y_test==1])[::-1]
    elif method == 'ae':
        idx = np.argsort(scores_ae[Y_test==1])[::-1]
    elif method == 'aexad_conv':
        idx = np.argsort(scores_aexad_conv[Y_test == 1])[::-1]

    print(idx[:5])


    plt.title(f'{method}')
    for i in range(5):
        plt.subplot(5, 6, (i*6)+1)
        plot_image(X_test[Y_test==1][idx[i]])
        plt.subplot(5, 6, (i*6)+2)
        plot_heatmap(htmaps_dev[Y_test==1][idx[i]].reshape(1, X_test.shape[-2], X_test.shape[-1]))
        plt.subplot(5, 6, (i*6)+3)
        plot_heatmap(htmaps_fcdd[Y_test==1][idx[i]])
        plt.subplot(5, 6, (i*6)+4)
        plot_heatmap(htmaps_ae[Y_test==1][idx[i]])
        plt.subplot(5, 6, (i*6)+5)
        plot_heatmap(htmaps_aexad_conv[Y_test==1][idx[i]])
        plt.subplot(5, 6, (i*6)+6)
        #plot_image(np.sqrt(htmaps_aexad_conv[Y_test == 1][idx[i]])+X_test[Y_test==1][idx[i]])
        #print((np.sqrt(htmaps_aexad_conv[Y_test == 1][idx[i]])+X_test[Y_test==1][idx[i]]).max())
        #plt.colorbar()
        #plt.subplot(5, 8, (i*8)+7)
        #plot_heatmap((np.sqrt(htmaps_aexad_conv[Y_test == 1][idx[i]])+X_test[Y_test==1][idx[i]])-X_test[Y_test==1][idx[i]])
        #plt.subplot(5, 8, (i*8)+8)
        intensity = GT_test[Y_test==1][idx[i]]*X_test[Y_test==1][idx[i]]
        plt.title(np.unique(intensity).sum())
        plot_heatmap(GT_test[Y_test==1][idx[i]])

    plt.show()

def plot_results(X_test, Y_test, GT_test, htmaps, ids):
    plt.figure(figsize=(9,15))
    for i in range(5):
        plt.subplot(5, 3, (i * 3) + 1)
        plt.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.imshow(X_test[ids[i]])
        plt.subplot(5, 3, (i * 3) + 2)
        plt.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.imshow(htmaps[ids[i]], cmap='jet', vmin=htmaps[ids[i]].min(), vmax=htmaps[ids[i]].max())
        plt.xlabel(f'AUC: {round(Xauc(GT_test[ids[i]], htmaps[ids[i]]), 4)}')
        #plt.colorbar()
        plt.subplot(5, 3, (i * 3) + 3)
        plt.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.imshow(GT_test[ids[i]], cmap='jet')
    plt.show()


def plot_top_flop(hts, gts, imgs, auc_ht, auc_sc, c, at, path, names):
    # Plot top 5 and flop 5 hatmaps for the test set
    ord_id = np.argsort(auc_ht)
    tp = np.append(ord_id[:5], ord_id[-5:])

    assert len(hts) == len(names)

    nrows = len(hts)+2

    fig, ax = plt.subplots(nrows, 10, figsize=(30, nrows*3+1))
    fig.suptitle(f'Class: {c} - Anomaly type: {at}', fontsize=30)
    fig.supxlabel(f'AUC: {round(auc_sc, 4)} - Average AUC pixel-wise: {round(auc_ht.mean(), 4)}', fontsize=30)

    for i in range(10):
        # Top
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])
        ax[0, i].imshow(imgs[tp[i]])
        for nht in range(len(hts)):
            ax[nht + 1, i].set_xticks([])
            ax[nht + 1, i].set_yticks([])
            if i == 0:
                ax[nht + 1, i].set_ylabel(names[nht], fontdict={'fontsize':24})
            im = ax[nht+1, i].imshow(hts[nht][tp[i]])
            #plt.colorbar(im, ax=ax[nht+1, i])
        ax[nrows - 1, i].set_xticks([])
        ax[nrows - 1, i].set_yticks([])
        ax[nrows - 1, i].set_xlabel(f'AUC: {round(auc_ht[tp[i]], 4)}', fontsize=20)
        ax[nrows - 1, i].imshow(gts[tp[i]])

    plt.savefig(path)

def plot_data(images, gts, model, id):
    rec = model(torch.from_numpy(images[id:id+1])).detach().cpu().numpy()[0]
    plt.figure()
    plt.imshow(images[id].transpose(1, 2, 0))
    plt.figure()
    plt.imshow(rec.transpose(1, 2, 0))
    plt.figure()
    plt.imshow(gts[id])
    plt.figure()
    plt.imshow(((rec - images[id]) ** 2).mean(axis=0), cmap='jet')


def plot_data_scales(images, gts, model, id, scales, show_auc = True):
    rec = model(torch.from_numpy(images[id:id+1])).detach().cpu().numpy()[0]
    ht = ((rec - images[id]) ** 2).mean(axis=0)
    plt.subplot(1, len(scales)+2, 1)
    plt.imshow(images[id].transpose(1, 2, 0))
    plt.axis('off')
    for i in range(len(scales)):
        s = scales[i]
        sigma = get_filter_sigma(ht, scale=s)
        ht_f = gaussian_filter(ht, sigma=sigma, truncate=3)
        plt.subplot(1, len(scales) + 2, i+2)
        plt.title(f'scale={s}')
        plt.imshow(ht_f, cmap='jet')
        if(show_auc):
            auc = Xauc(gts[id], ht_f)
            print(auc)
            plt.xlabel(f'AUC: {np.round(auc, 4)}')
        plt.xticks([])
        plt.yticks([])
        #plt.axis('off')
    plt.subplot(1, len(scales)+2, len(scales)+2)
    plt.title('GT')
    plt.imshow(gts[id], cmap='gray')
    plt.axis('off')
    plt.show()


def rad_from_lines(gt, d_thres=3):
    nc = gt.shape[1]
    gtrow = gt.cumsum(axis=1) * gt
    drow = gtrow[np.where(gtrow[:, 0:nc - 1] > gtrow[:, 1:nc])]
    drow = np.concatenate((drow, gtrow[gtrow[:, nc - 1] > 0, nc - 1]))
    drow = drow[drow > d_thres]
    rrow = 0.5 * drow.mean()
    nr = gt.shape[0]
    gtcol = gt.cumsum(axis=0) * gt
    dcol = gtcol[np.where(gtcol[0:nr - 1, :] > gtcol[1:nr, :])]
    dcol = np.concatenate((dcol, gtcol[nr - 1, gtcol[nr - 1, :] > 0]))
    dcol = dcol[dcol > d_thres]
    rcol = 0.5 * dcol.mean()
    return min(rcol, rrow)


def scores_fabrizio():
    return None

def score_neighs(ht, rad=1):
    sc = 0
    for i in range(ht.shape[0]):
        for j in range(ht.shape[1]):
            l = max(0, j - rad)
            r = min(ht.shape[1], j + rad + 1)
            u = max(0, i - rad)
            d = min(ht.shape[0], i + rad + 1)
            w = np.sum(ht[u:d, l:r]) - ht[i, j]
            w = w /((r-l)*(d-u)-1)
            sc = sc + ht[i, j]*w
    return sc


def scores_neighs(hts, rad=1):
    scores = np.zeros(hts.shape[0])
    for i in range(hts.shape[0]):
        ht = hts[i]
        scores[i] = score_neighs(ht, rad)
    return scores

def get_filter_sigma(ht, thres=0.5, scale=0.25):
    if ht.max() > 1:
        warn.warn('Error: heatmap not in [0,1]')
    binht = np.where(ht >= thres, 1, 0)
    rad = rad_from_lines(binht)
    return max(rad * scale, 1)


def blurred_htmaps(hts, thres=0.5, scale=0.25):
    hts_b = np.empty_like(hts)
    for i in range(hts.shape[0]):
        sigma = get_filter_sigma(hts[i], thres=thres, scale=scale)
        hts_b[i] = gaussian_filter(hts[i], sigma=sigma)

    return hts_b

def plot_ht_auc(df, save_path=None, datt='', MS=100, LW=4, FS = 23, aexad_v='aexad_blur'):
    plt.figure(figsize=(10, 10))
    plt.scatter(df['fcdd'], df[aexad_v], marker='+', s=MS, linewidths=LW, label='FCDD')
    plt.scatter(df['devnet'], df[aexad_v], marker='+', s=MS, linewidths=LW, label='DEV-NET')
    #plt.scatter(df['ae'], df['aexad_blur'], marker='+', s=MS, linewidths=LW, label='AE')
    # plt.axis('square')
    plt.xlim((0.3, 1))
    plt.ylim((0.3, 1))
    plt.plot([0.3, 1], [0.3, 1], '-k')
    plt.legend(loc='lower right')
    plt.title(datt, fontsize=FS)
    plt.ylabel('AE-XAD [AUC]', fontsize=FS)
    plt.xticks(fontsize=FS)
    plt.yticks(fontsize=FS)
    plt.grid()

    if save_path is not None:
        plt.savefig(save_path)


def plot_htmaps(ids, X_test, GT_test, ht_aexad_blur, ht_ae_res, ht_fcdd_res, ht_devnet, title, save_path=None):
    fig, axs = plt.subplots(len(ids), 6, gridspec_kw={'wspace': 0, 'hspace': 0})
    fig.suptitle(title)
    for i in range(len(ids)):
        title = 'Image' if i == 0 else ''
        axs[i][0].set_title(title)
        axs[i][0].imshow(X_test[ids[i]])
        axs[i][0].set_aspect('equal')
        axs[i][0].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        title = 'GT' if i == 0 else ''
        axs[i][1].set_title(title)
        axs[i][1].imshow(GT_test[ids[i]])
        axs[i][1].set_aspect('equal')
        axs[i][1].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        title = 'AE-XAD' if i == 0 else ''
        axs[i][2].set_title(title)
        axs[i][2].imshow(ht_aexad_blur[ids[i]], cmap='jet')
        axs[i][2].set_aspect('equal')
        axs[i][2].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        title = 'AE' if i == 0 else ''
        axs[i][3].set_title(title)
        axs[i][3].imshow(ht_ae_res[ids[i]], cmap='jet')
        axs[i][3].set_aspect('equal')
        axs[i][3].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        title = 'FCDD' if i == 0 else ''
        axs[i][4].set_title(title)
        axs[i][4].imshow(ht_fcdd_res[ids[i]], cmap='jet')
        axs[i][4].set_aspect('equal')
        axs[i][4].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        title = 'DEVNET' if i == 0 else ''
        axs[i][5].set_title(title)
        axs[i][5].imshow(ht_devnet[ids[i]], cmap='jet')
        axs[i][5].set_aspect('equal')
        axs[i][5].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plots_std_bin(j, ids_anomalies, x, gt, hs1, hs2, hs3, hs4, num_stds, dataset, c=None):
    '''
    Plot function for heatmaps binarized according to different sigma thresholds
    '''
    print(j)
    id  = ids_anomalies[j]
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(wspace=0, hspace=0)
    for i in range(len(hs1)):
        plt.subplot(6, len(hs1), 1 + i)
        plt.imshow(x[id], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.ylabel('IMG')
        #plt.title('IMG')
        plt.subplot(6, len(hs1), len(hs1) + 1 + i)
        plt.imshow(gt[id], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.ylabel('GT')
        #plt.title('GT')
        plt.subplot(6, len(hs1), len(hs1)*2 + 1 + i)
        plt.imshow(hs1[i][id], cmap='jet')
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.ylabel('AE-XAD')
        #plt.title(f'AE-XAD')
        plt.subplot(6, len(hs1), len(hs1)*3 + 1 + i)
        plt.imshow(hs2[i][id], cmap='jet')
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.ylabel('AE')
        plt.subplot(6, len(hs1), len(hs1)*4 + 1 + i)
        plt.imshow(hs3[i][id], cmap='jet')
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.ylabel('FCDD')
        plt.subplot(6, len(hs1), len(hs1)*5 + 1 + i)
        plt.imshow(hs4[i][id], cmap='jet')
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.ylabel('BGAD')
        plt.xlabel(rf'$\sigma={num_stds[i]}$')

    pr = perimeter(gt[id])
    ar = area(gt[id])
    plt.suptitle(r"$\mathbf{P^2/A}$: " + f"{(pr ** 2) / ar:.3f}", fontsize=10, fontweight='bold')

    if c is None:
        name = f'PLOT/{dataset}_final_f2/fig_{j}_std.jpg'
    else:
        name = f'PLOT/{dataset}_final_f2/{c}/fig_{j}_std.jpg'
    plt.savefig(name)
    plt.close()


def plots_hts_ids(ids_anomalies, x, gt, hs1, hs2, hs3, hs4, c_ids, an_type):
    '''
    Plot function for heatmaps binarized according to different sigma thresholds
    '''

    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(wspace=0, hspace=0)

    for i in range(len(c_ids)):
        j = c_ids[i]
        id = ids_anomalies[j]

        plt.subplot(6, len(c_ids), 1 + i)
        plt.imshow(x[id], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.ylabel('IMG')
        #plt.title('IMG')
        plt.subplot(6, len(c_ids), len(c_ids) + 1 + i)
        plt.imshow(gt[id], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.ylabel('GT')
        #plt.title('GT')
        plt.subplot(6, len(c_ids), len(c_ids)*2 + 1 + i)
        plt.imshow(hs1[i][id], cmap='jet')
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.ylabel('AE-XAD')
        #plt.title(f'AE-XAD')
        plt.subplot(6, len(c_ids), len(c_ids)*3 + 1 + i)
        plt.imshow(hs2[i][id], cmap='jet')
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.ylabel('AE')
        plt.subplot(6, len(c_ids), len(c_ids)*4 + 1 + i)
        plt.imshow(hs3[i][id], cmap='jet')
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.ylabel('FCDD')
        plt.subplot(6, len(c_ids), len(c_ids)*5 + 1 + i)
        plt.imshow(hs4[i][id], cmap='jet')
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.ylabel('BGAD')

    name = f'PLOT/{an_type}/fig_anoms.jpg'
    plt.savefig(name)
    plt.close()