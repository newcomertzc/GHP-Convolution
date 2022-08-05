"""Unlike torchvision.transforms which are aimed at transforming data with label, this transforms are aimed at giving 
label to data according to the transforms they have gone through. So most of these transforms take only image instead 
of (image, label) as input.

In addition, to avoid padding in transformed image (it's harmful to Image Forensics), these transfroms don't guarantee
a non-none result.

For convience, these transforms only support RGB images (or RGBA images) and Gray images stored as array.
"""
from . import functional
from .transforms import *