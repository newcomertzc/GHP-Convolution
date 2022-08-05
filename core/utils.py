import os
import random
import pickle
import numpy as np

import torch
import torch.nn.functional as F

from copy import deepcopy

from .common_types import *


__all__ = [
    'capitalize_name', 'keep_dir_valid', 'get_dir_name', 'get_base_name', 'path_join',
    'set_seed', 'use_deterministic_algorithms', 'zero_pad', 'random_zero_pad', 'calc_accuracy', 'calc_mIoU',
    'label_smooth', 'one_hot', 'predict_multiclass', 'predict_binary', 'LabelSmoothingCrossEntropyLoss', 
    'FunctionExecutor', 'MetricKeeper'
]


special_words = {
    'net': 'Net',
    'next': 'NeXt',
    'vgg': 'VGG',
}


def capitalize_name(name: str) -> str:
    name = name.lower()
    for key, val in special_words.items():
        name = name.replace(key, val)
    name = name.capitalize()
    return name


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    

def use_deterministic_algorithms(use: bool) -> None:
    torch.backends.cudnn.benchmark = not use
    torch.backends.cudnn.deterministic = use


def keep_dir_valid(path: str) -> None:
    if not os.path.exists(path):
        os.mkdir(path)


def get_dir_name(path: str, last_folder: bool = True) -> str:
    path = path.replace('/', '\\')
    dir_name = os.path.dirname(path)
    if last_folder:
        dir_name = dir_name.split('\\')[-1]
    return dir_name
        
        
def get_base_name(path: str, no_suffix: bool = True) -> str:
    base_name = os.path.basename(path)
    if no_suffix and '.' in base_name:
        base_name = base_name[:base_name.rindex('.')]
    return base_name


def path_join(*paths: Any) -> str:
    path = os.path.join(*paths)
    
    return path


def zero_pad(image, target_size, offset):
    h, w = image.shape[:2]
    tH, tW = target_size

    if h > tH or w > tW:
        return None

    i, j = offset
    
    if len(image.shape) == 3:
        new_image = np.zeros((tH, tW, image.shape[2]), dtype=image.dtype)
    else:
        new_image = np.zeros((tH, tW), dtype=image.dtype)
        
    new_image[i: i + h, j: j + w] = image
    return new_image


def random_zero_pad(image, target_size):
    h, w = image.shape[:2]
    tH, tW = target_size

    if h > tH or w > tW:
        return None, None

    i = np.random.randint(tH - h + 1)
    j = np.random.randint(tW - w + 1)
    
    if len(image.shape) == 3:
        new_image = np.zeros((tH, tW, image.shape[2]), dtype=image.dtype)
    else:
        new_image = np.zeros((tH, tW), dtype=image.dtype)
        
    new_image[i: i + h, j: j + w] = image
    return new_image, (i, j)


def label_smooth(label: Tensor, num_classes: int, alpha: float = 0.1) -> Tensor:
    """Label smoothing algorithm.

    Args:
        label (torch.Tensor): Original label.
        num_classes (int): Total number of classes.
        alpha (float, optional): 
            A float in [0.0, 1.0]. Specifies the amount of smoothing. Defaults to 0.1.

    Returns:
        torch.Tensor: Smoothed label.
    """
    new_label = torch.ones_like(label) * alpha / (num_classes - 1)
    new_label[torch.where(label == 1)] = 1 - alpha
    
    return new_label


def one_hot(targets: Tensor, num_classes: int) -> Tensor:
    if len(targets.size()) == 1:
        new_targets = F.one_hot(targets, num_classes)
        
        return new_targets
    elif len(targets.size()) == 3:
        new_targets = F.one_hot(targets, num_classes)
        new_targets = new_targets.permute(0, 3, 1, 2)
        
        return new_targets
    else:
        raise ValueError(f"the size {targets.size()} of targets is invalid!")


def predict_multiclass(output: Tensor) -> ndarray:
    """Predict multiclass results.

    Args:
        output (torch.Tensor)

    Returns:
        numpy.ndarray: Prediction.
    """

    preds = torch.argmax(output, dim=1)
    return preds.cpu().numpy()


def predict_binary(output: Tensor, threshold: float = 0.5) -> ndarray:
    """Predict binary results.

    Args:
        output (torch.Tensor)
        threshold (float): [description]. Defaults to 0.5.

    Returns:
        numpy.ndarray: Prediction.
    """
    probs = torch.sigmoid(output)
    preds = (probs > threshold).to(torch.int64)
    return preds.cpu().numpy()


def multiclass_to_binary(output: Tensor) -> Tensor:
    """Convert multiclass results to binary results.

    Args:
        output (torch.Tensor)

    Returns:
        numpy.ndarray: Prediction.
    """

    output[:, 1] = output[:, 1:].max(dim=1).values
    output = output[:, :2]
    return output


class LabelSmoothingCrossEntropyLoss(Module):
    def __init__(self, label_smoothing: float = 0.1) -> None:
        """CrossEntropyLoss with label smoothing.
        
        Introduced by et al. at Rethinking the Inception Architecture for Computer Vision.
        Source code from: https://github.com/seominseok0429/label-smoothing-visualization-pytorch

        Args:
            label_smoothing (float, optional): Defaults to 0.1.
        """
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.label_smoothing = label_smoothing
        
    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        confidence = 1. - self.label_smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.label_smoothing * smooth_loss
        return loss.mean()


class FunctionExecutor:
    def __init__(self, func: Callable, kwargs: dict) -> None:
        self.func = func
        self.kwargs = kwargs
        
    def execute(self) -> Any:
        return self.func(**self.kwargs)
    

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
    

class MetricKeeper:
    def __init__(self) -> None:
        self.defaults = {}
    
    def add_metric(self, name: str, default: Any = None) -> None:
        self.defaults[name] = deepcopy(default)
        setattr(self, name, default)
        
    def reset_metric(self, name: str) -> None:
        setattr(self, name, self.defaults[name])
    
    def all_metrics(self) -> dict:
        return vars(self)

    def reset(self) -> None:
        for name, default in self.defaults.items():
            setattr(self, name, default)