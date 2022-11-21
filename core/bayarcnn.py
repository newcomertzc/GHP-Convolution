import torch

from .base import BaseBayarCNN
from .common_types import *


__all__ = ['BayarCNN', 'BayarCNN_box', 'BayarCNN_GHP']


class BayarCNN(BaseBayarCNN):
    name = 'BayarCNN'
    
    def __init__(
        self,
        num_classes: int
    ) -> None:
        """CNN architecture from 'A Deep Learning Approach To Universal Image 
        Manipulation Detection Using A New Convolutional Layer'.

        Args:
            num_classes (int): Total number of classes.
            in_channels (int, optional): Number of input channels. Defaults to 1.
        """
        super(BayarCNN, self).__init__(num_classes)

    def forward(self, x: Tensor) -> Tensor:
        self.norm_weight()
        x = super(BayarCNN, self).forward(x)
        
        return x

    def norm_weight(self):
        h, w = self.convRes.kernel_size
        cH, cW = h // 2,  w // 2

        s = self.convRes.weight.data.sum(dim=[2, 3])
        s -= self.convRes.weight.data[:, :, cH, cW]

        s = s.view(s.size(0), s.size(1), 1, 1)
        self.convRes.weight.data /= s
        self.convRes.weight.data[:, :, cH, cW] = -1
        

class BayarCNN_box(BayarCNN):
    name = 'BayarCNN-box'
    
    def __init__(
        self,
        num_classes: int
    ) -> None:
        """CNN architecture from 'A Deep Learning Approach To Universal Image 
        Manipulation Detection Using A New Convolutional Layer'.

        Args:
            num_classes (int): Total number of classes.
            in_channels (int, optional): Number of input channels. Defaults to 1.
        """
        super(BayarCNN_box, self).__init__(num_classes)
        self.convRes.weight.data = torch.ones_like(self.convRes.weight) / 24
        self.convRes.weight.data[:, :, 2, 2] = -1


class BayarCNN_GHP(BaseBayarCNN):
    name = 'BayarCNN-GHP'
    
    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.01,
        penalty: str = 'L1',
        reduction: str = 'sum'
    ) -> None:
        """A variant of BayarCNN. Its first layer is replaced with GHP Convolution.

        Args:
            num_classes (int): Total number of classes.
            in_channels (int, optional): Number of input channels. Defaults to 1.
            alpha (float, optional): Penalty factor for regularization loss.
                Defaults to 0.01.
            penalty (str, optional): Regularization technique used to calculate 
                regularization loss. 'L1' or 'L2'. Defaults to 'L1'.
        """
        super(BayarCNN_GHP, self).__init__(num_classes)
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

    def calc_reg_loss(self) -> Tensor:
        reg_funcs = {
            'L1': torch.abs,
            'L2': torch.square
        }
        gather_funcs = {
            'sum': torch.sum,
            'mean': torch.mean
        }
        reg_func = reg_funcs[self.penalty]
        gather_func = gather_funcs[self.reduction]
        reg_loss = self.alpha * torch.sum(reg_func(gather_func(self.convRes.weight, dim=[2, 3])))
        
        return reg_loss