import torch.nn as nn

from copy import deepcopy
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from ..backbone import *
from ...base import BaseFCN_VGG, BaseSegNetwork
from ...common_types import *


__all__ = [
    'FCN_VGG_32s', 'FCN_VGG_16s', 'FCN_VGG_8s', 'SegResNet', 'FCN_ResNet', 'DeepLabV3_ResNet',
    'FCNHead', 'DeepLabHead',
]
    
    
class FCN_VGG_32s(BaseFCN_VGG):
    def __init__(
        self, 
        num_classes: int,
        in_channels: int = 3,
        backbone_func: Callable = VGG, 
        backbone_kwargs: dict = {},
        size_adaptive: bool = True
    ) -> None:
        """FCN-VGG-32s.
        """
        super(FCN_VGG_32s, self).__init__(
            num_classes, in_channels, backbone_func, backbone_kwargs)
        if size_adaptive:
            self.upsample_x32 = nn.UpsamplingBilinear2d()
        else:
            self.upsample_x32 = nn.UpsamplingBilinear2d(scale_factor=32)
        self.name += '-32s'
        
    def forward(self, x: Tensor) -> Tensor:
        if self.upsample_x32.scale_factor is None:
            self.upsample_x32.size = (x.size(2), x.size(3))
        
        result = OrderedDict()
        x = self.backbone(x)
        x = self.classifier(x)
        
        x = self.upsample_x32(x)
        result['out'] = x
        
        return result
    

class FCN_VGG_16s(BaseFCN_VGG):
    def __init__(
        self, 
        num_classes: int,
        in_channels: int = 3,
        backbone_func: Callable = VGG, 
        backbone_kwargs: dict = {},
        size_adaptive: bool = True
    ) -> None:
        """FCN-VGG-16s.
        """
        super(FCN_VGG_16s, self).__init__(
            num_classes, in_channels, backbone_func, backbone_kwargs)
        self.upsample_x2 = nn.UpsamplingBilinear2d(scale_factor=2)
        if size_adaptive:
            self.upsample_x16 = nn.UpsamplingBilinear2d()
        else:
            self.upsample_x16 = nn.UpsamplingBilinear2d(scale_factor=16)
        self.name += '-16s'
        self.classifier_pool4 = nn.Conv2d(512, num_classes, 1)
        
    def forward(self, x: Tensor) -> Tensor:
        if self.upsample_x16.scale_factor is None:
            self.upsample_x16.size = (x.size(2), x.size(3))
        pool4_index = self.pool_index[3]
        
        result = OrderedDict()
        x_pool4 = self.backbone[: pool4_index + 1](x)
        x = self.backbone[pool4_index + 1:](x_pool4)
        x = self.classifier(x)
        
        # upsample to stride 16 and fuse
        x = self.upsample_x2(x)
        x_pool4 = self.classifier_pool4(x_pool4)
        # crop_size = ((x_pool4.size(2) - x.size(2)) // 2, 
        #              (x_pool4.size(3) - x.size(3)) // 2)
        crop_size = (6, 6)
        
        x_pool4 = x_pool4[:, :,
                          crop_size[0]: crop_size[0] + x.size(2),
                          crop_size[1]: crop_size[1] + x.size(3)]
        x += x_pool4
        
        x = self.upsample_x16(x)
        result['out'] = x
        
        return result


class FCN_VGG_8s(BaseFCN_VGG):
    def __init__(
        self, 
        num_classes: int,
        in_channels: int = 3,
        backbone_func: Callable = VGG, 
        backbone_kwargs: dict = {},
        size_adaptive: bool = True
    ) -> None:
        """FCN-VGG-8s.
        """
        super(FCN_VGG_8s, self).__init__(
            num_classes, in_channels, backbone_func, backbone_kwargs)
        self.upsample_x2 = nn.UpsamplingBilinear2d(scale_factor=2)
        if size_adaptive:
            self.upsample_x8 = nn.UpsamplingBilinear2d()
        else:
            self.upsample_x8 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.name += '-8s'
        self.classifier_pool3 = nn.Conv2d(256, num_classes, 1)
        self.classifier_pool4 = nn.Conv2d(512, num_classes, 1)
        
    def forward(self, x: Tensor) -> Tensor:
        if self.upsample_x8.scale_factor is None:
            self.upsample_x8.size = (x.size(2), x.size(3))
        pool3_index = self.pool_index[2]
        pool4_index = self.pool_index[3]
        
        result = OrderedDict()
        x_pool3 = self.backbone[: pool3_index + 1](x)
        x_pool4 = self.backbone[pool3_index + 1: pool4_index + 1](x_pool3)
        x = self.backbone[pool4_index + 1:](x_pool4)
        x = self.classifier(x)
        
        # upsample to stride 16 and fuse
        x = self.upsample_x2(x)
        x_pool4 = self.classifier_pool4(x_pool4)
        # crop_size = ((x_pool4.size(2) - x.size(2)) // 2, 
        #              (x_pool4.size(3) - x.size(3)) // 2)
        crop_size = (6, 6)
        
        x_pool4 = x_pool4[:, :,
                          crop_size[0]: crop_size[0] + x.size(2),
                          crop_size[1]: crop_size[1] + x.size(3)]
        x += x_pool4
        
        # upsample to stride 32 and fuse
        x = self.upsample_x2(x)
        x_pool3 = self.classifier_pool3(x_pool3)
        # crop_size = ((x_pool3.size(2) - x.size(2)) // 2, 
        #              (x_pool3.size(3) - x.size(3)) // 2)
        crop_size = (12, 12)
        
        x_pool3 = x_pool3[:, :,
                          crop_size[0]: crop_size[0] + x.size(2),
                          crop_size[1]: crop_size[1] + x.size(3)]
        x += x_pool3
        
        x = self.upsample_x8(x)
        result['out'] = x
        
        return result


class SegResNet(BaseSegNetwork):
    def __init__(
        self,
        num_classes: int,
        classifier_func: Callable,
        in_channels: int = 3,
        backbone_func: Callable = ResNet, 
        backbone_kwargs: dict = {}, 
        aux_loss: bool = False,
        replace_stride_with_dilation: Optional[List[bool]] = [False, True, True]
    ) -> None:
        """Segmentation ResNet.
        
        Args:
            classifier_func: (Callable): FCNHead or DeepLabHead.
            replace_stride_with_dilation: (List, optional)
                replace_stride_with_dilation    stride
                --------
                [False, True, True]             8s
                [False, False, True]            16s
                [False, False, False]           32s
        """
        backbone_kwargs = deepcopy(backbone_kwargs)
        backbone_kwargs['model_kwargs'] = backbone_kwargs.get('model_kwargs', {})
        backbone_kwargs['model_kwargs']['replace_stride_with_dilation'] = replace_stride_with_dilation
        super(SegResNet, self).__init__(
            num_classes, classifier_func, in_channels, 
            backbone_func, backbone_kwargs, aux_loss)
        
        model_map = {
            FCNHead: 'FCN',
            DeepLabHead: 'DeepLabV3'
        }
        self.name = f"{model_map.get(classifier_func, 'Seg')}-{self.name}"
        if replace_stride_with_dilation is None:
            self.name += f"-32s"
        else:
            dilation = replace_stride_with_dilation
            stride = 2 ** sum(dilation)
            self.name += f"-{32 // stride}s"  
        
    def analyse_model(self, model: Module) -> tuple:
        out_layer = 'layer4'
        out_inplanes = 2048
        aux_layer = 'layer3'
        aux_inplanes = 1024
        
        return out_layer, out_inplanes, aux_layer, aux_inplanes
    

class FCN_ResNet(SegResNet):
    def __init__(
        self,
        num_classes: int,
        classifier_func: Callable = FCNHead,
        in_channels: int = 3,
        backbone_func: Callable = ResNet, 
        backbone_kwargs: dict = {}, 
        aux_loss: bool = False,
        replace_stride_with_dilation: Optional[List[bool]] = [False, True, True]
    ) -> None:
        super(FCN_ResNet, self).__init__(
            num_classes, classifier_func, in_channels, 
            backbone_func, backbone_kwargs, aux_loss, replace_stride_with_dilation)
        

class DeepLabV3_ResNet(SegResNet):
    def __init__(
        self,
        num_classes: int,
        classifier_func: Callable = DeepLabHead,
        in_channels: int = 3,
        backbone_func: Callable = ResNet, 
        backbone_kwargs: dict = {}, 
        aux_loss: bool = False,
        replace_stride_with_dilation: Optional[List[bool]] = [False, True, True]
    ) -> None:
        super(DeepLabV3_ResNet, self).__init__(
            num_classes, classifier_func, in_channels, 
            backbone_func, backbone_kwargs, aux_loss, replace_stride_with_dilation)