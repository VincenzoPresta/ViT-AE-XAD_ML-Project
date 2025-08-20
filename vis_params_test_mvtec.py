import os
import torch

from torch import Tensor
from kornia import gaussian_blur2d
from tools.evaluation_metrics import Xauc
import numpy as np

import torchvision.transforms as transforms
import PIL.Image as Image

def __global_norm(imgs: Tensor, qu: int, ref: Tensor = None) -> Tensor:
    """
    Applies a global normalization of tensor, s.t. the highest value of the complete tensor is 1 and
    the lowest value is >= zero. Uses a non-linear normalization based on quantiles as explained in the appendix
    of the paper.
    :param imgs: images tensor
    :param qu: quantile used
    :param ref: if this is None, normalizes w.r.t. to imgs, otherwise normalizes w.r.t. to ref.
    """
    ref = ref if ref is not None else imgs
    imgs.sub_(ref.min())
    ref = ref.sub(ref.min())
    quantile = ref.reshape(-1).kthvalue(int(qu * ref.reshape(-1).size(0)))[0]  # qu% are below that
    imgs.div_(quantile)  # (1 - qu)% values will end up being out of scale ( > 1)
    plosses = imgs.clamp(0, 1)  # clamp those
    return plosses


def __local_norm(imgs: Tensor, qu: int, ref: Tensor = None) -> Tensor:
    """
    Applies a local normalization of tensor, s.t. the highest value of each element (dim=0) in the tensor is 1 and
    the lowest value is >= zero. Uses a non-linear normalization based on quantiles as explained in the appendix
    of the paper.
    :param imgs: images tensor
    :param qu: quantile used
    """
    imgs.sub_(imgs.reshape(imgs.size(0), -1).min(1)[0][(...,) + (None,) * (imgs.dim() - 1)])
    quantile = imgs.reshape(imgs.size(0), -1).kthvalue(
        int(qu * imgs.reshape(imgs.size(0), -1).size(1)), dim=1
    )[0]  # qu% are below that
    imgs.div_(quantile[(...,) + (None,) * (imgs.dim() - 1)])
    imgs = imgs.clamp(0, 1)  # clamp those
    return imgs


if __name__ == '__main__':
    test = 'gauss_kernel'
    classes = [0, 1, 3]

    transform_gt = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((448, 448), Image.NEAREST),
        transforms.Grayscale()
    ])

    if test == 'gauss_kernel':
        sigmas = [1, 2, 4, 8, 16, 32]
        kss = [5, 9, 17, 33, 65]
        seeds = [40]

        rets = np.empty((len(classes), len(sigmas), len(kss), len(seeds)))

        for nc in range(3):
            c = classes[nc]
            print('Class: ', c)
            for r in range(len(seeds)):
                seed = seeds[r]
                print('Seed: ', seed)
                path = os.path.join('results', 'f_1', 'mvtec', str(c), str(seed))
                Y_test = np.load(open(os.path.join(path, 'labels.npy'), 'rb'))
                GT_test = np.load(open(os.path.join(path, 'gt.npy'), 'rb'))[Y_test==1]
                GT_test_r = np.empty((GT_test.shape[0], 448, 448))
                for i in range(len(GT_test)):
                    GT_test_r[i] = np.array(transform_gt(GT_test[i])) / 255.0
                htmaps_aexad = np.load(open(os.path.join(path, 'aexad_htmaps_conv.npy'), 'rb'))[Y_test==1]
                htmaps_aexad = htmaps_aexad.mean(axis=1)[:, np.newaxis]

                for s in range(len(sigmas)):
                    sigma = sigmas[s]
                    print('Sigma: ', sigma)
                    for ks in range(len(kss)):
                        kernel = kss[ks]
                        print('Kernel: ', kernel)

                        htmaps_f = gaussian_blur2d(torch.from_numpy(htmaps_aexad), kernel_size=(kernel, kernel),
                                                   sigma=(sigma, sigma))
                        aucs = np.empty(htmaps_f.shape[0])
                        for i in range(len(aucs)):
                            aucs[i] = Xauc(GT_test_r[i], htmaps_f[i])

                        rets[nc, s, ks, r] = aucs.mean()
                        print(aucs.mean())

        np.save(open('filters_aucs_gauss_mvtec.npy', 'wb'), rets)
        print(rets.mean(axis=-1))

    elif test == 'norm':
        etas = [0.85, 0.90, 0.95, 0.97, 0.99]
        seeds = [29, 38, 40, 57, 75]

        rets = np.empty((len(classes), len(sigmas), len(kss), len(seeds)))

        for nc in range(3):
            c = classes[nc]
            print('Class: ', c)
            for r in range(len(seeds)):
                seed = seeds[r]
                print('Seed: ', seed)
                path = os.path.join('results', 'f_1', 'fmnist', str(c), str(seed))
                Y_test = np.load(open(os.path.join(path, 'labels.npy'), 'rb'))
                GT_test = np.load(open(os.path.join(path, 'gt.npy'), 'rb'))[Y_test == 1]
                GT_test_r = np.empty((GT_test.shape[0], 448, 448))
                for i in range(len(GT_test)):
                    GT_test_r[i] = np.array(transform_gt(GT_test[i])) / 255.0
                htmaps_aexad = np.load(open(os.path.join(path, 'aexad_htmaps_conv.npy'), 'rb'))[Y_test == 1]
                htmaps_aexad = htmaps_aexad.mean(axis=1)[:, np.newaxis]

                for e in range(len(etas)):
                    eta = etas[e]
                    print('Eta: ', eta)

                    htmaps_f = __local_norm(torch.from_numpy(htmaps_aexad), eta)
                    aucs = np.zeros(htmaps_f.shape[0])
                    for i in range(len(aucs)):
                        aucs[i] = Xauc(GT_test[i], htmaps_f[i])

                    rets[c, e, 0, r] = aucs.mean()
                    print(aucs.mean())

                    htmaps_f = __global_norm(torch.from_numpy(htmaps_aexad), eta)
                    aucs = np.zeros(htmaps_f.shape[0])
                    for i in range(len(aucs)):
                        aucs[i] = Xauc(GT_test[i], htmaps_f[i])

                    rets[nc, e, 1, r] = aucs.mean()
                    print(aucs.mean())

        np.save(open('filters_aucs_norm_mvtec.npy', 'wb'), rets)
        print(rets.mean(axis=-1))
