import os
import random
import pickle
import numpy as np

import torch
import torch.nn.functional as F

from copy import deepcopy
from ptflops import get_model_complexity_info

from .common_types import *


__all__ = [
    'capitalize_name', 'keep_dir_valid', 'get_dir_name', 'get_base_name', 'path_join', 'str_to_bool', 'zero_pad', 'random_zero_pad', 
    'set_seed', 'use_deterministic_algorithms', 'extract_weights', 'freeze_weights', 'label_smooth', 'one_hot', 
    'predict_multiclass', 'predict_binary', 'calc_accuracy', 'calc_mIoU', 'multiclass_to_binary', 'show_model', 'print_inconsistent_kwargs',
    'get_JPEG_stat', 'LabelSmoothingCrossEntropyLoss', 'FunctionExecutor', 'MetricKeeper'
]


special_words = {
    'net': 'Net',
    'next': 'NeXt',
    'vgg': 'VGG',
    'Vgg': 'VGG',
}


def str_to_bool(s) -> bool:
    if s == '0' or s == 'False':
        return False
    return True


def capitalize_name(name: str) -> str:
    name = name.capitalize()
    for key, val in special_words.items():
        name = name.replace(key, val)
    return name


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    

def use_deterministic_algorithms(use: bool) -> None:
    torch.backends.cudnn.benchmark = not use
    torch.backends.cudnn.deterministic = use
    

def extract_weights(
    weights: OrderedDict, 
    desired_layer_name: Optional[str] = None, 
    target_layer_name: Optional[str] = None
) -> OrderedDict:
    """Extract the weights of desired layer (and modify its name).
    """
    weights = weights.copy()
    if desired_layer_name is None:
        for name in list(weights.keys()): # avoid OrderedDict mutating during iteration
            new_name = target_layer_name + '.' + name
            weights[new_name] = weights.pop(name)
    else:
        for name in list(weights.keys()):
            if desired_layer_name in name:
                if target_layer_name is not None:
                    new_name = name.replace(desired_layer_name, target_layer_name)
                    weights[new_name] = weights.pop(name)
            else:
                del weights[name]
            
    return weights


def freeze_weights(
    model: Module, 
    freeze: bool = True,
    layer_name: Optional[str] = None
) -> None:
    if layer_name is None:
        for param in model.parameters():
            param.requires_grad = not freeze
    else:
        for name, param in model.named_parameters():
            if layer_name in name:
                param.requires_grad = not freeze


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


def label_smooth(label: Tensor, alpha: float = 0.1) -> Tensor:
    """Label smoothing algorithm.

    Args:
        label (torch.Tensor): Original label.
        alpha (float, optional): 
            A float in [0.0, 1.0]. Specifies the smoothing factor. Defaults to 0.1.

    Returns:
        torch.Tensor: Smooth label.
    """
    smooth_label = label * (1 - alpha) + alpha / 2
    
    return smooth_label


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
    

def calc_accuracy(cmatrix: ndarray) -> float:
    num_classes = cmatrix.shape[0]
    
    t = np.sum([cmatrix[class_idx, class_idx] 
             for class_idx in range(num_classes)])
    total = np.sum(cmatrix)
    
    return t / total


def calc_mIoU(cmatrix: ndarray) -> float:
    num_classes = cmatrix.shape[0]
    
    IoUs = []
    for class_idx in range(num_classes):
        Inter = cmatrix[class_idx, class_idx]
        Union = sum(cmatrix[class_idx, :]) + sum(cmatrix[:, class_idx]) - Inter
        IoU = Inter / Union
        IoUs.append(IoU)
    
    return np.mean(IoUs)


def show_model(model: Module, input_size: tuple) -> None:
    macs, params = get_model_complexity_info(model, input_size, print_per_layer_stat=False)
    print(f"architecture:\n{model}")
    print(f"complexity: {macs} | params: {params}")
    

def get_JPEG_stat(stat_path: str) -> tuple:
    with open(stat_path, 'rb') as f:
        stat = pickle.load(f)
    
    quality = dict(Counter(stat['quality']).most_common(30))
    if -1 in quality:
        del quality[-1]
    quality_values, quality_probs = list(quality.keys()), list(quality.values())
    
    subsampling = Counter(stat['subsampling'])
    subsampling_probs = [subsampling[key] for key in ['4:4:4', '4:2:2', '4:2:0']]
    
    return quality_values, quality_probs, subsampling_probs
            

def print_inconsistent_kwargs(kwargs_cur: dict, kwargs_prev: dict, prefix: str = '') -> None:
    if kwargs_prev != kwargs_cur:
        if prefix == '':
            print('found inconsistent kwargs:')
        for key, value in kwargs_prev.items():
            new_value = kwargs_cur.get(key)
            
            if isinstance(value, dict) and new_value is not None:
                print_inconsistent_kwargs(value, new_value, prefix = f"{key}.")
            else:
                if value != new_value:
                    print(f" {prefix}{key}: {value} -> {new_value}")
                

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
