import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

#from modeling.net import SemiADNet
from competitors.deviation.modeling.net import SemiADNet
#from datasets import mvtecad
from competitors.deviation.datasets import mvtecad
import cv2
import os
import argparse
from modeling.layers import build_criterion
from utils import aucPerformance
from scipy.ndimage.filters import gaussian_filter

np.seterr(divide='ignore',invalid='ignore')
#fmap_block = list()
#grad_block = list()

def backward_hook(module, grad_in, grad_out):
    #global grad_block
    grad_block.append(grad_out[0].detach())

def farward_hook(module, input, output):
    #global fmap_block
    fmap_block.append(output)

def convert_to_grayscale(im_as_arr):
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    if im_max > 0:
        grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im

def show_cam_on_image(img, mask, label, out_dir, name):

    img1 = img.copy()
    img[:, :, 0] = img1[:, :, 2]
    img[:, :, 1] = img1[:, :, 1]
    img[:, :, 2] = img1[:, :, 0]

    label = label.swapaxes(0, 1).swapaxes(1, 2)

    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam = np.concatenate((img, cam), axis=1)
    cam = np.concatenate((cam, label), axis=1)

    print('PATH: ', os.path.join(out_dir, "_cam_" + name + ".jpg"))
    #path_cam_img = os.path.join(out_dir, args.classname + "_cam_" + name + ".jpg")
    path_cam_img = os.path.join(out_dir, "_cam_" + name + ".jpg")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))

def get_heatmaps_and_scores(model, test_loader, cuda):
    print('----- CALCOLO HEATMAP')
    model.eval()

    global fmap_block
    global grad_block
    fmap_block = list()
    grad_block = list()
    model.feature_extractor.net.layer4[1].conv2.register_forward_hook(farward_hook)
    model.feature_extractor.net.layer4[1].conv2.register_backward_hook(backward_hook)


    #seg_label = list()
    #outliers_cam = list()
    #labels = list()
    #outlier_scores = list()
    outputs = list()
    fmaps = list()
    grads = list()
    seg_label = list()
    outliers_cam = list()
    input = list()
    outlier_scores = list()
    labels = list()

    tbar = tqdm(range(len(test_loader.dataset)))
    for i in tbar:
        model.zero_grad()
        sample = test_loader.dataset.getitem(i)

        inputs = sample['image'].view(1, 3, test_loader.dataset.args.img_size, test_loader.dataset.args.img_size)
        input.append(np.array(sample['raw_image']))
        if cuda:
            inputs = inputs.cuda()
        inputs.requires_grad = True
        output = model(inputs)
        outlier_scores.append(output.data.cpu().numpy()[0][0])
        output.backward()
        labels.append(sample['label'])

        grad = inputs.grad
        grad_temp = convert_to_grayscale(grad.cpu().numpy().squeeze(0))
        grad_temp = grad_temp.squeeze(0)
        grad_temp = gaussian_filter(grad_temp, sigma=4)
        print(grad_temp.shape)
        #grad_temp = np.array(Image.fromarray(np.uint8(grad_temp * 255)).resize(test_loader.dataset.original_shape)) / 255
        outliers_cam.append(grad_temp)


        seg_label.append(np.array(sample['seg_label']))
        fmap_block = list()
        grad_block = list()


    for i, (cam, raw, label) in enumerate(zip(outliers_cam, input, seg_label)):
        print("Sto salvando l'immagine")
        raw = np.float32(cv2.resize(np.array(raw), (test_loader.dataset.args.img_size, test_loader.dataset.args.img_size))) / 255
        #label = cv2.resize(label, (test_loader.dataset.args.img_size, test_loader.dataset.args.img_size)) / 255
        show_cam_on_image(raw, cam, label, '.', str(i))


    return outliers_cam, outlier_scores, seg_label, labels
