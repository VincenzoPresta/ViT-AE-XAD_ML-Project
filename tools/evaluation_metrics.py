import numpy as np
from sklearn.metrics import roc_auc_score,jaccard_score, precision_recall_curve

def Xauc(ground_truth,heatmap):
    '''
    :param ground_truth: binary matrix indicating whether each pixel of an image is actually anomalous (1) or normal (0)
    :param heatmap: real value matrix containing the predicted anomaly score of each pixel
    :return: Area Under the ROC Curve relative to the anomaly score of the pixels of a fixed image
    '''
    return roc_auc_score(ground_truth.flatten(),heatmap.flatten())

def IoU(ground_truth,heatmap):
    '''
    :param ground_truth: binary matrix indicating whether each pixel of an image is actually anomalous (1) or normal (0)
    :param heatmap: binary matrix containing the anomaly prediction for each pixel
    :return: I/U, where I and U are respectively the intersection and the union of the anomalous pixels of the ground-truth and the predicted heatmap
    '''
    return jaccard_score(ground_truth.flatten(),heatmap.flatten())

def IoU_avg(ground_truth,heatmap):
    '''
    :param ground_truth: binary matrix indicating whether each pixel of an image is actually anomalous (1) or normal (0)
    :param heatmap: real value matrix containing the predicted anomaly score of each pixel
    :return: Average IoU varying the threshold of the heatmap in order to make it binary
    '''
    summ = 0
    for t in heatmap.flatten():
        binary_heatmap = (heatmap > t).astype(int)
        summ = summ + IoU(ground_truth,binary_heatmap)
    return summ/ground_truth.size

def IoU_max(ground_truth,heatmap):
    '''
    :param ground_truth: binary matrix indicating whether each pixel of an image is actually anomalous (1) or normal (0)
    :param heatmap: real value matrix containing the predicted anomaly score of each pixel
    :return: Maximum IoU varying the threshold of the heatmap in order to make it binary
    '''
    max = 0
    for t in heatmap.flatten():
        binary_heatmap = (heatmap > t).astype(int)
        curr = IoU(ground_truth,binary_heatmap)
        if max<curr:
            max = curr
    return max


def average_precision(y, score):
    precision, _, _ = precision_recall_curve(y.flatten(), score.flatten())
    return precision.mean()

def average_recall(y, score):
    _, recall, _ = precision_recall_curve(y.flatten(), score.flatten())
    return recall.mean()


def precision(y, y_pred):
    tp = np.sum((y.flatten()==1) & (y_pred.flatten()==1))
    return tp / y_pred.sum()

def recall(y, y_pred):
    tp = np.sum((y.flatten()==1) & (y_pred.flatten()==1))
    return tp / y.sum()

def iou(y, y_pred):
    intersection = np.sum((y.flatten()==1) & (y_pred.flatten()==1))
    union = np.sum((y.flatten()==1) | (y_pred.flatten()==1))
    return intersection / union
    
    
def area(gt):
    return gt.sum()


def perimeter(gt):
    rows = gt.shape[0]
    cols = gt.shape[1]
    area_pixels = np.where(gt == 1)
    area_pixels_r = area_pixels[0]
    area_pixels_c = area_pixels[1]
    perim = area(gt)
    for p in range(len(area_pixels[0])):
        c = area_pixels_c[p]
        r = area_pixels_r[p]
        if r>0 and c>0 and r<rows-1 and c<cols-1:
            perim = perim - gt[r-1, c]*gt[r+1, c]*gt[r, c-1]*gt[r, c+1]
    return perim


def Xaucs(GT, hs):
    Xaucs = np.zeros(hs.shape[0])
    for i in range(hs.shape[0]):
        h = hs[hs[i]]
        Xaucs[i] = roc_auc_score(GT[hs[i]].flatten().astype(int), h.flatten())
    return Xaucs