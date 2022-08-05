import numpy as np

from .common_types import *


__all__ = ['calc_accuracy', 'calc_mIoU']


def calc_accuracy(cmatrix):
    num_classes = cmatrix.shape[0]
    
    t = np.sum([cmatrix[class_idx, class_idx] 
             for class_idx in range(num_classes)])
    total = np.sum(cmatrix)
    
    return t / total


def calc_mIoU(cmatrix):
    num_classes = cmatrix.shape[0]
    
    IoUs = []
    for class_idx in range(num_classes):
        Inter = cmatrix[class_idx, class_idx]
        Union = sum(cmatrix[class_idx, :]) + sum(cmatrix[:, class_idx]) - Inter
        IoU = Inter / Union
        IoUs.append(IoU)
    
    return np.mean(IoUs)