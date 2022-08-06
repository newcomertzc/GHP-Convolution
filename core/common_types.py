from numpy import ndarray
from torch import Tensor, dtype as torch_dtype
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.optim.optimizer import Optimizer
from collections import Counter, OrderedDict
from PIL.ImageFile import ImageFile
from typing import Callable, Optional, Union, List, Any, Generator


__all__ = [
    'ndarray', 'Tensor', 'torch_dtype', 'Module', 'ImageFile', 'Counter', 'OrderedDict', 'Dataset', 'DataLoader', 
    'Optimizer', 'Callable', 'Optional', 'Union', 'List', 'Any', 'Generator', '_valid_data_types'
]

_valid_data_types = Optional[Union[ndarray, dict]]