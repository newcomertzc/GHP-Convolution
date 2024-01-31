import torch
import torch.nn as nn

from .base import *
from .common_types import *


__all__ = ['PreprocConv2d', 'PreprocGHPConv2d']


class PreprocConv2d(BaseModule):
    name = 'Conv'
    
    def __init__(
        self, 
        in_channels: int = 1,
        out_channels: int = 12, 
        kernel_size: int = 5,
        bias: bool = True,
        depthwise: bool = True,
        norm_layer: Callable[..., Module] = nn.Identity,
        norm_layer_kwargs: dict = {},
        activ_layer: Callable[..., Module] = nn.Identity,
        activ_layer_kwargs: dict = {},
    ) -> None:
        """Plain Convolution for data preprocessing.

        Args:
            in_channels (int, optional): Number of input channels. Default to 1.
            out_channels (int, optional): Number of output channels. Default to 12.
            depthwise (bool, optional): If True, set the groups of the convolutional 
                layer to in_channels. Defaults to True.
        """
        super(PreprocConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding = kernel_size // 2, 
                              groups = in_channels if depthwise else 1, bias = bias)
        self.norm = norm_layer(out_channels, **norm_layer_kwargs)
        self.activ = activ_layer(**activ_layer_kwargs)
        
        if isinstance(self.norm, nn.BatchNorm2d):
            self.name += 'BN'    
        if not isinstance(self.activ, nn.Identity):
            self.name += f"{self.activ.__class__.__name__}"
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activ(x)
        
        return x
    
    def get_params_label(self) -> List[str]:
        params_label = super(PreprocConv2d, self).get_params_label()
        
        if self.conv.in_channels != 1:
            params_label.append(f"c{self.conv.in_channels}")
        if self.conv.out_channels != 12:
            params_label.append(f"w{self.conv.out_channels}")
        if self.conv.bias is None:
            params_label.append('nb')
        if self.conv.kernel_size[0] != 5:
            params_label.append(f"k{self.conv.kernel_size[0]}")
            
        return params_label
    
    
class PreprocGHPConv2d(PreprocConv2d):
    name = 'GHPConv'
    
    def __init__(
        self, 
        in_channels: int = 1,
        out_channels: int = 12, 
        kernel_size: int = 5,
        bias: bool = True,
        depthwise: bool = True,
        alpha: Optional[float] = None,
        penalty: str = 'L2',
        norm_layer: Callable[..., Module] = nn.Identity,
        norm_layer_kwargs: dict = {},
        activ_layer: Callable[..., Module] = nn.Identity,
        activ_layer_kwargs: dict = {},
    ) -> None:
        """GHP Convolution for data preprocessing.
        For ResNet, the optimal penalty and alpha is L1-norm and 0.01 (or L2-norm and 3.0). 

        Args:
            in_channels (int, optional): Number of input channels. Default to 1.
            out_channels (int, optional): Number of output channels. Default to 12.
            depthwise (bool, optional): If True, set the groups of the convolutional 
                layer to in_channels. Default to True.
            alpha (float, optional): Penalty factor for regularization loss. Default to 3. 
            penalty (str, optional): Regularization technique used to calculate 
                regularization loss. 'L1' or 'L2'. Default to 'L2'.
        """
        super(PreprocGHPConv2d, self).__init__(
            in_channels, out_channels, kernel_size, bias, depthwise, norm_layer, norm_layer_kwargs, activ_layer, activ_layer_kwargs)
        
        valid_penalty = {'L1', 'L2'}
        if penalty not in valid_penalty:
            raise ValueError(f"penalty must be one of {valid_penalty},"
                             f" but got penalty='{penalty}'")
        
        default_alpha = {
            'L1': 0.01,
            'L2': 3.0
        }
        if alpha is None:
            alpha = default_alpha[penalty]
            
        self.penalty = penalty
        self.alpha = alpha
            
    def calc_reg_loss(self) -> Tensor:
        reg_funcs = {
            'L1': torch.abs,
            'L2': torch.square
        }
        reg_func = reg_funcs[self.penalty]
        reg_loss = self.alpha * torch.sum(
            reg_func(
                torch.sum(self.conv.weight, dim=[2, 3])))
        
        return reg_loss
    
    def get_params_label(self) -> List[str]:
        params_label = super(PreprocGHPConv2d, self).get_params_label()
        
        params_label.append(f"{self.penalty}")
        params_label.append(f"a{self.alpha}")
            
        return params_label
