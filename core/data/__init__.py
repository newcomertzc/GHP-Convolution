"""The dataloader in Pytorch can only load a fixed number of items, but the items in my dataset might be None.
As a result, the batch_size will varies in training, which is unfavorable for conv acceleration.

To fix it, I have customized a generator dataloader, which will keep a batch_size of fixed number when items 
might be None.

You can use torchvision.transforms after a non-none judgement in after_transform.
"""
from .data import *
from . import functional