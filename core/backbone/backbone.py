import torch.nn as nn

from torchvision import models as torchvision_models

from .. import convnext
from ..base import InheritModule
from ..common_types import *


__all__ = [
    'ResNet', 'VGG', 'ConvNeXt', 'resnext101_64x4d'
]


class ResNet(InheritModule, torchvision_models.ResNet):
    def __init__(
        self, 
        num_classes: int,
        in_channels: int = 3,
        model_func: Callable = torchvision_models.resnet50, 
        model_kwargs: dict = {}
    ) -> None:
        """ResNet (ResNeXt).
        
        Args:
            num_classes (int): Total number of classes.
            in_channels (int, optional): Number of input channels. Defaults to 3. 
        """
        InheritModule.__init__(self, num_classes, in_channels, model_func, model_kwargs)
    
    def customize_model(
        self, 
        num_classes: int, 
        in_channels: int, 
        model_func: Callable, 
        model_kwargs: dict = {}
    ) -> Module:
        model = model_func(num_classes=num_classes, **model_kwargs)
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        return model


def resnext101_64x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 64
    kwargs['width_per_group'] = 4
    return torchvision_models.resnet._resnet(
        'resnext101_64x4d', torchvision_models.resnetBottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


class VGG(InheritModule, torchvision_models.VGG):
    def __init__(
        self, 
        num_classes: int,
        in_channels: int = 3,
        model_func: Callable = torchvision_models.vgg13,
        model_kwargs: dict = {}, 
    ) -> None:
        """VGG.
        
        Args:
            num_classes (int): Total number of classes.
            in_channels (int, optional): Number of input channels. Defaults to 3. 
        """
        InheritModule.__init__(self, num_classes, in_channels, model_func, model_kwargs)

    def customize_model(
        self, 
        num_classes: int, 
        in_channels: int, 
        model_func: Callable, 
        model_kwargs: dict = {}
    ) -> Module:
        model = model_func(num_classes=num_classes, **model_kwargs)
        model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        
        return model


class ConvNeXt(InheritModule, convnext.ConvNeXt):
    def __init__(
        self, 
        num_classes: int,
        in_channels: int = 3,
        model_func: Callable = convnext.convnext_tiny,
        model_kwargs: dict = {}, 
    ) -> None:
        """ConvNeXt.
        
        Args:
            num_classes (int): Total number of classes.
            in_channels (int, optional): Number of input channels. Defaults to 3. 
        """
        InheritModule.__init__(self, num_classes, in_channels, model_func, model_kwargs)

    def customize_model(
        self, 
        num_classes: int, 
        in_channels: int, 
        model_func: Callable, 
        model_kwargs: dict = {}
    ) -> Module:
        model = model_func(in_chans=in_channels, num_classes=num_classes, **model_kwargs)
        
        return model