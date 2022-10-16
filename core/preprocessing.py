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
        norm_layer: Callable = nn.Identity,
        norm_layer_kwargs: dict = {},
        activ_layer: Module = nn.Identity()
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
        self.activ = activ_layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activ(x)
        
        return x
    
    def get_params_label(self) -> List[str]:
        params_label = super(PreprocConv2d, self).get_params_label()
        
        if self.conv.in_channels != 1:
            params_label.append(f"{self.conv.in_channels}c")
        if self.conv.out_channels != 12:
            params_label.append(f"w{self.conv.out_channels}")
            
        if self.conv.kernel_size[0] != 5:
            params_label.append(f"s{self.conv.kernel_size[0]}")
        
        if isinstance(self.norm, nn.BatchNorm2d):
            params_label.append('BN')
        elif isinstance(self.norm, nn.LayerNorm):
            params_label.append('LN')
            
        if not isinstance(self.activ, nn.Identity):
            params_label.append(self.activ.__class__.__name__)
            
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
        alpha: float = 3.0,
        penalty: str = 'L2',
        reduction: str = 'sum',
        norm_layer: Callable = nn.Identity,
        norm_layer_kwargs: dict = {},
        activ_layer: Module = nn.Identity()
    ) -> None:
        """GHP Convolution for data preprocessing.

        Args:
            in_channels (int, optional): Number of input channels. Default to 1.
            out_channels (int, optional): Number of output channels. Default to 12.
            depthwise (bool, optional): If True, set the groups of the convolutional 
                layer to in_channels. Defaults to True.
            alpha (float, optional): Penalty factor for regularization loss. Defaults to 3.0.
            penalty (str, optional): Regularization technique used to calculate 
                regularization loss. 'L1' or 'L2'. Defaults to 'L2'.
        """
        super(PreprocGHPConv2d, self).__init__(
            in_channels, out_channels, kernel_size, bias, depthwise, norm_layer, norm_layer_kwargs, activ_layer)
        valid_penalty = {'L1', 'L2'}
        valid_reduction = {'sum', 'mean'}
        if penalty not in valid_penalty:
            raise ValueError(f"penalty must be one of {valid_penalty},"
                             f" but got penalty='{penalty}'")
        if reduction not in valid_reduction:
            raise ValueError(f"reduction must be one of {valid_reduction},"
                             f" but got reduction='{reduction}'")
        self.alpha = alpha
        self.penalty = penalty
        self.reduction = reduction
    
    def calc_reg_loss(self) -> torch.Tensor:
        reg_funcs = {
            'L1': torch.abs,
            'L2': torch.square
        }
        sum_funcs = {
            'sum': torch.sum,
            'mean': torch.mean
        }
        reg_func = reg_funcs[self.penalty]
        sum_func = sum_funcs[self.reduction]
        reg_loss = self.alpha * torch.sum(reg_func(sum_func(self.conv.weight, dim=[2, 3])))
        
        return reg_loss
    
    def get_params_label(self) -> List[str]:
        params_label = super(PreprocGHPConv2d, self).get_params_label()
        
        params_label.append(f"{self.penalty}")
        if self.alpha != 1.0:
            params_label.append(f"a{self.alpha}")
            
        return params_label