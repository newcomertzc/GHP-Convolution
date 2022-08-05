import torch
import torch.nn as nn

from torchvision import models as torchvision_models

from .utils import capitalize_name
from .common_types import *


__all__ = [
    'BaseModule', 'NormModule', 'InheritModule', 'NormIdentity', 'PreprocIdentity', 
    'BaseBayarCNN', 'BaseFCN_VGG', 'BaseSegNetwork'
]


class BaseModule(Module):
    name = None
    kwargs = None
    
    def get_params_label(self) -> List[str]:
        return []
    
    def calc_reg_loss(self) -> Union[float, Tensor]:
        return 0.0
    
    def copy_vars(self, vars_dict: dict) -> None:
        for key, val in vars_dict.items():
            setattr(self, key, val)
    
    
class NormModule(BaseModule):
    def __init__(self, size: int) -> None:
        super(NormModule, self).__init__()
        

class InheritModule(BaseModule):
    def __init__(
        self, 
        num_classes: int,
        in_channels: int,
        model_func: Callable, 
        model_kwargs: dict = {}, 
    ) -> None:
        """Inherit the model while add some new functions.
        """
        BaseModule.__init__(self)
        model = self.customize_model(
            num_classes, in_channels, model_func, model_kwargs)
        self.name = capitalize_name(model_func.__name__)
        self.copy_vars(vars(model))
        
    def customize_model(
        self, 
        num_classes: int,
        in_channels: int,
        model_func: Callable, 
        model_kwargs: dict = {}
    ) -> Module:
        """Should be overridden by all subclasses.
        """
        raise NotImplementedError


class NormIdentity(NormModule, nn.Identity):
    """Placeholder.
    """
    pass


class PreprocIdentity(BaseModule, nn.Identity):
    """Placeholder.
    """
    pass


class BaseBayarCNN(BaseModule):
    def __init__(
        self,
        num_classes: int
    ) -> None:
        super(BaseBayarCNN, self).__init__()
        self.convRes = nn.Conv2d(1, 12, 5, padding=2)
        self.conv1 = nn.Conv2d(12, 64, 7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 48, 5, padding=2)
        self.fc1 = nn.Linear(37632, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.5)
        self.lrn = nn.LocalResponseNorm(5, alpha=0.0001*5, k=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.convRes(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.lrn(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.lrn(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class BaseFCN_VGG(BaseModule):
    def __init__(
        self, 
        num_classes: int,
        in_channels: int,
        backbone_func: Callable, 
        backbone_kwargs: dict = {}
    ) -> None:
        super(BaseFCN_VGG, self).__init__()
        backbone = backbone_func(in_channels=in_channels, num_classes=num_classes, **backbone_kwargs)
        self.backbone = backbone.features
        self.backbone[0].padding = (100, 100)
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(True),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, 1)
        )
        self.name = f"FCN-{backbone.name}"
        
    def forward(self, x: Tensor) -> Tensor:
        """Should be overridden by all subclasses.
        """
        raise NotImplementedError


class BaseSegNetwork(BaseModule):
    def __init__(
        self, 
        num_classes: int,
        classifier_func: Callable,
        in_channels: int,
        backbone_func: Callable,
        backbone_kwargs: dict,
        aux_loss: bool = False,
    ) -> None:
        super(BaseSegNetwork, self).__init__()
        backbone = backbone_func(in_channels=in_channels, num_classes=num_classes, **backbone_kwargs)
        out_layer, out_inplanes, aux_layer, aux_inplanes = self.analyse_model(backbone)
        
        return_layers = {out_layer: 'out'}
        if aux_loss:
            return_layers[aux_layer] = 'aux'
            
        self.backbone = torchvision_models._utils.IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.classifier = classifier_func(out_inplanes, num_classes)
        self.aux_classifier = None if not aux_loss else classifier_func(aux_inplanes, num_classes)
        
        self.upsample = nn.UpsamplingBilinear2d()
        self.name = backbone.name
    
    def forward(self, x):
        self.upsample.size = (x.size(2), x.size(3))
        features = self.backbone(x)
        
        result = OrderedDict()
        x = features['out']
        x = self.classifier(x)
        x = self.upsample(x)
        result['out'] = x

        if self.aux_classifier is not None:
            x = features['aux']
            x = self.aux_classifier(x)
            x = self.upsample(x)
            result['aux'] = x

        return result
        
    def analyse_model(self, model: BaseModule) -> tuple:
        """Should be overridden by all subclasses.
        """
        raise NotImplementedError