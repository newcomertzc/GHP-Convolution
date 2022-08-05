from .base import *
from .common_types import *


__all__ = ['IMDNetwork']


class IMDNetwork(BaseModule):
    @staticmethod
    def build_with_kwargs(
        backbone_func: Callable,
        backbone_kwargs: dict = {},
        preprocessing_func: Callable = PreprocIdentity,
        preprocessing_kwargs: dict = {},
        name: Optional[str] = None
    ) -> BaseModule:
        kwargs = locals()
        preprocessing = preprocessing_func(**preprocessing_kwargs)
        backbone = backbone_func(**backbone_kwargs)
        
        instance = IMDNetwork(backbone, preprocessing, name)
        instance.kwargs = kwargs
        return instance
    
    def __init__(
        self, 
        backbone: BaseModule, 
        preprocessing: BaseModule = PreprocIdentity(),
        name: Optional[str] = None
    ) -> None:
        """Image Manipulation Detection Networks.
        
        Customized model is expected to inherit BaseModule when used as backbone. 
        """
        super(IMDNetwork, self).__init__()
        self.preprocessing = preprocessing
        self.backbone = backbone
        self.kwargs = None
        self.name = self.get_default_name() if name is None else name
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.preprocessing(x)
        x = self.backbone(x)
        
        return x
    
    def calc_reg_loss(self):
        reg_loss = self.preprocessing.calc_reg_loss()
        reg_loss += self.backbone.calc_reg_loss()
        
        return reg_loss
    
    def get_default_name(self):
        preproc_name = self.preprocessing.name
        backbone_name = self.backbone.name
        preproc_params_label = self.preprocessing.get_params_label()
        backbone_params_label = self.backbone.get_params_label()
                
        subnames = []
        if preproc_name is not None:
            subnames.append(preproc_name)
        subnames.append(backbone_name)
        subnames.extend(backbone_params_label)
        subnames.extend(preproc_params_label)
        
        name = '-'.join(subnames)
        
        return name