import os
import numpy as np

from AE_architectures import Shallow_Autoencoder, Deep_Autoencoder, Conv_Autoencoder, Conv_Deep_Autoencoder, \
    PCA_Autoencoder, Conv_Deep_Autoencoder_Norm, Conv_Deep_Autoencoder_v2
from aexad_script import Trainer
import torch

from tools.create_dataset import mvtec_personalized, load_dataset, mvtec, mvtec_only_one
from torchvision import transforms
from PIL import Image

from tools.utils import blurred_htmaps


def load_model(path, dim, latent_dim=None, AE_type='conv_deep'):
    if AE_type == 'shallow':
        model = Shallow_Autoencoder(dim, np.prod(dim), latent_dim)
    # deep
    elif AE_type == 'deep':
        model = Deep_Autoencoder(dim, flat_dim=np.prod(dim), intermediate_dim=256, latent_dim=latent_dim)
    # conv
    elif AE_type == 'conv':
        model = Conv_Autoencoder(dim)

    elif AE_type == 'conv_deep':
        model = Conv_Deep_Autoencoder(dim)

    elif AE_type == 'conv_deep_v2':
        model = Conv_Deep_Autoencoder_v2(dim)

    elif AE_type == 'conv_deep_norm':
        model = Conv_Deep_Autoencoder_Norm(dim)

    elif AE_type == 'pca':
        model = PCA_Autoencoder(np.prod(dim), np.prod(dim), latent_dim)
    else:
        raise Exception('Model not yet implemented')


    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def load_data_all(path, data_path):
    _, _, X_test, Y_test, _, GT_test = load_dataset(data_path)

    ht_aexad = np.load(open(os.path.join(path, 'aexad_htmaps_conv.npy'), 'rb')).mean(axis=1).astype(np.float32)
    sc_aexad = np.load(open(os.path.join(path, 'aexad_scores_conv.npy'), 'rb'))
    ht_ae = np.load(open(os.path.join(path, 'mse_htmaps_conv.npy'), 'rb')).mean(axis=1).astype(np.float32)
    sc_ae = np.load(open(os.path.join(path, 'mse_scores_conv.npy'), 'rb'))
    ht_devnet = np.load(open(os.path.join(path, 'deviation_htmaps.npy'), 'rb'))
    sc_devnet = np.load(open(os.path.join(path, 'deviation_scores.npy'), 'rb'))
    ht_fcdd = np.load(open(os.path.join(path, 'fcdd_htmaps.npy'), 'rb'))[:, 0]
    sc_fcdd = np.load(open(os.path.join(path, 'fcdd_scores.npy'), 'rb'))

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(X_test.shape[1:3]),
        transforms.ToTensor(),
    ])

    ht_aexad_res = np.empty((GT_test.shape[0], GT_test.shape[1], GT_test.shape[2]))
    ht_ae_res = np.empty((GT_test.shape[0], GT_test.shape[1], GT_test.shape[2]))
    ht_fcdd_res = np.empty(GT_test.shape[:3])
    for i in range(ht_aexad.shape[0]):
        ht_aexad_res[i] = transform(ht_aexad[i])
        ht_ae_res[i] = transform(ht_ae[i])
        ht_fcdd_res[i] = transform(ht_fcdd[i])

    ht_aexad_res_blur = blurred_htmaps(ht_aexad_res, scale=1.0)
    GT_test = GT_test[:, :, :, 0]

    return ht_aexad_res, ht_aexad_res_blur, ht_fcdd_res, ht_devnet, ht_ae_res, sc_aexad, sc_fcdd, sc_devnet, sc_ae, X_test, GT_test, Y_test



def load_data_aexad(root, ht_file, score_file, shape, model_file=None, lm=False, AE_type='conv_deep'):

    ht_aexad = np.load(open(os.path.join(root, ht_file), 'rb')).swapaxes(1,2).swapaxes(2,3) # 21/11/24 mean sostituito con sum
    #ht_aexad = ht_aexad.sum(axis=1).astype(np.float32)
    #ht_aexad = (ht_aexad * 255).astype(np.uint8)
    sc_aexad = np.load(open(os.path.join(root, score_file), 'rb'))

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(shape[2:], Image.NEAREST),
        transforms.PILToTensor(),
    ])

    ##gt_aexad = np.empty((GT_test.shape[0], 448, 448))
    ##imgs_aexad = np.empty((GT_test.shape[0], 3, 448, 448))
    ##ht_aexad = ht_aexad.mean(axis=1)
    ##ht_aexad = ht_aexad.swapaxes(1, 2).swapaxes(2, 3)
    #ht_aexad_res = np.empty(shape)
    #for i in range(ht_aexad.shape[0]):
    #    #gt_aexad[i] = transform(GT_test[i])[0]
    #    #imgs_aexad[i] = transform(X_test[i])
    #    ht_aexad_res[i] = transform(ht_aexad[i]).numpy()

    #ht_aexad_res = ht_aexad_res / 255.

    if lm:
        model = load_model(os.path.join(root, model_file), (3, ht_aexad.shape[1], ht_aexad.shape[2]), latent_dim=None, AE_type=AE_type)
    else:
        model = None

    return ht_aexad.sum(axis=-1), sc_aexad, model # _res.sum(axis=1)


def load_data_fcdd(root, c, na, s, shape, at=None):

    if at is None:
        fcdd_p = os.path.join(root, str(c), str(s), str(na))
    else:
        fcdd_p = os.path.join(root, str(c), str(s), str(at), str(na))

    ht_fcdd = np.load(open(os.path.join(fcdd_p, 'fcdd_htmaps.npy'), 'rb'))[:, 0]
    sc_fcdd = np.load(open(os.path.join(fcdd_p, 'fcdd_scores.npy'), 'rb'))

    transform_fcdd = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(shape[1:]),
        transforms.ToTensor(),
    ])

    ##gt_fcdd = np.empty((GT_test.shape[0], 448, 448))
    ##imgs_fcdd = np.empty((GT_test.shape[0], 3, 448, 448))
    #ht_fcdd_res = np.empty(shape)
    #for i in range(ht_fcdd.shape[0]):
    #    #gt_fcdd[i] = transform_fcdd(GT_test[i])[0]
    #    #imgs_fcdd[i] = transform_fcdd(X_test[i])
    #    ht_fcdd_res[i] = transform_fcdd(ht_fcdd[i]).numpy()

    return ht_fcdd, sc_fcdd


def load_data_devnet(root, c, na, s, at=0):

    if at is None:
        devnet_p = os.path.join(root, str(c), str(s), str(na))
    else:
        devnet_p = os.path.join(root, str(c), str(s), str(at), str(na))

    ht_devnet = np.load(open(os.path.join(devnet_p, 'deviation_htmaps.npy'), 'rb'))
    sc_devnet = np.load(open(os.path.join(devnet_p, 'deviation_scores.npy'), 'rb'))


    return ht_devnet, sc_devnet