import numpy as np
import cv2 as cv

from os import getpid
from PIL import Image
from torch.nn.modules.utils import _pair
from torchvision.transforms import Normalize, ToTensor

from .functional import *
from ..utils import *
from ..common_types import *


__all__ = [
    'get_input_transform', 'Remain', 'PrintInfo', 'SelectChannel', 'RandomHorizontalFlip', 'RandomVerticalFlip',
    'PreserveMaskValue', 'TransformBinaryMask', 'Resize', 'Compose', 'Parallel', 'PILToArray', 'RGBAToRGB',
    'ToRGB', 'ToGray', 'ToContiguousArray', 'RandomResize', 'RandomResize_discrete', 'RandomScale', 'RandomScale_discrete',
    'RandomRotate', 'RandomRotate_discrete', 'RandomMedianFilter', 'RandomBoxFilter', 'RandomGaussianFilter',
    'RandomAWGN', 'RandomAWGN_discrete', 'PoissonNoise', 'RandomImpulseNoise', 'RandomImpulseNoise_discrete',
    'Crop', 'RandomCrop', 'RandomPatch', 'RandomNPatches', 'AllPatches', 'SelectPatches', 'RandomTransform',
    'JPEGCompress', 'RandomJPEGCompress', 'WEBPCompress', 'RandomWEBPCompress', 'Normalize', 'ToTensor'
]


def get_input_transform(input_type: str) -> tuple:
    """Get input transform according to input type.
    
    The mean and std is calculated according to https://github.com/pytorch/vision/pull/1965.

    Args:
        input_type (str): 'green', 'gray' or 'rgb'.

    Returns:
        tuple: (input_convert, input_normalize)
    """
    input_transforms = {
        'green': (
            Compose([ToRGB(), SelectChannel(1)]),
            Normalize(
                mean=[0.454],
                std=[0.220])),
        'gray': (
            ToGray(),
            Normalize(
                mean=[0.457],
                std=[0.221])),
        'rgb': (
            ToRGB(),
            Normalize(
                mean=[0.484, 0.454, 0.403],
                std=[0.225, 0.220, 0.220]))
    }

    return input_transforms[input_type]


class Remain:
    """Placeholder.
    """
    def __call__(self, x: Any) -> Any:
        return x
    
    
class PrintInfo:
    def __call__(self, x: _valid_data_types) -> _valid_data_types:
        print('-' * 3)
        if x is None:
            print('type: None')
            return None
        
        print(f"type: {type(x)}")
        if isinstance(x, dict):
            print(f"image size: {np.array(x['image']).shape}")
            print(f"mask size: {np.array(x['mask']).shape}")
        else:
            print(f"size: {x.shape}")
        return x


class SelectChannel:
    def __init__(self, channel_idx: int) -> None:
        self.channel_idx = channel_idx

    def __call__(self, x: _valid_data_types) -> _valid_data_types:
        if x is None:
            return None
        
        image = x['image'] if isinstance(x, dict) else x
        image = image[:, :, self.channel_idx]
        
        if isinstance(x, dict):
            x['image'] = image
        else:
            x = image
        return x


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p
        
    def __call__(self, x: _valid_data_types) -> _valid_data_types:
        if x is None:
            return None
        
        if isinstance(x, dict):
            if np.random.rand() < self.p:
                x['image'] = x['image'][:, ::-1]
                x['mask'] = x['mask'][:, ::-1]
        else:
            if np.random.rand() < self.p:
                x = x[:, ::-1]
        return x
    
    
class RandomVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p
        
    def __call__(self, x: _valid_data_types) -> _valid_data_types:
        if x is None:
            return None
        
        if isinstance(x, dict):
            if np.random.rand() < self.p:
                x['image'] = x['image'][::-1, :]
                x['mask'] = x['mask'][::-1, :]
        else:
            if np.random.rand() < self.p:
                x = x[:, ::-1]
        return x
    
    
class PreserveMaskValue:
    def __init__(self, transform: Callable, key: Any = 'mask'):
        """This transform preserve the values of mask.   
        """
        self.transform = transform
        self.key = key
        
    def __call__(self, x: _valid_data_types) -> _valid_data_types:
        if x is None:
            return None
        
        if self.key is None:
            mask = x
        else:
            mask = x[self.key]

        label = np.unique(mask)[-1]
        mask = np.array(mask / label, dtype=mask.dtype)
        
        if self.key is None:
            x = mask
        else:
            x[self.key] = mask
            
        x = self.transform(x)
        if self.key is None:
            return x * label
        else:
            x[self.key] = x[self.key] * label
            return x
        
        
class TransformBinaryMask:
    def __init__(self, key: Any = 'mask'):
        self.key = key
        
    def __call__(self, x: _valid_data_types) -> _valid_data_types:
        if x is None:
            return None
        
        if self.key is None:
            mask = x
        else:
            mask = x[self.key]

        mask = np.array(mask / 255, dtype=mask.dtype)
        if self.key is None:
            x = mask
        else:
            x[self.key] = mask
            
        return x
    
    
# TODO add type hints
    
class Resize:
    def __init__(self, size, interpolation=cv.INTER_LINEAR):
        self.size = _pair(size)
        self.interpolation = interpolation
        
    def __call__(self, x):
        if x is None:
            return None
        
        image = x['image'] if isinstance(x, dict) else x
        image = cv.resize(image, self.size[::-1], interpolation=self.interpolation)
    
        if isinstance(x, dict):
            x['image'] = image
            x['mask'] = cv.resize(x['mask'], self.size[::-1], interpolation=self.interpolation)
            
        else:
            x = image
        return x


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        if x is None:
            return None

        for transform in self.transforms:
            x = transform(x)

        return x


class Parallel:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        if x is None:
            return None

        output = []
        for transform in self.transforms:
            output.append(transform(x))
        return output


class PILToArray:
    def __call__(self, x):
        if x is None:
            return None
        
        image = x['image'] if isinstance(x, dict) else x
        image = np.array(image)
        
        if isinstance(x, dict):
            x['image'] = image
            x['mask'] = np.array(x['mask'])
        else:
            x = image
        return x


class RGBAToRGB:
    def __call__(self, x):
        if x is None:
            return None
        
        image = x['image'] if isinstance(x, dict) else x
        if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
            image = cv.cvtColor(image, cv.COLOR_RGBA2RGB)
        
        if isinstance(x, dict):
            x['image'] = image
        else:
            x = image
        return x 


class ToRGB:
    def __call__(self, x):
        if x is None:
            return None

        image = x['image'] if isinstance(x, dict) else x
        if len(image.shape) == 2:  # Gray
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv.cvtColor(image, cv.COLOR_RGBA2RGB)
        
        if isinstance(x, dict):
            x['image'] = image
        else:
            x = image
        return x


class ToGray:
    def __call__(self, x):
        if x is None:
            return None

        image = x['image'] if isinstance(x, dict) else x
        if len(image.shape) == 2:
            return x
        
        if image.shape[2] == 3:  # RGB
            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:  # RGBA
            image = cv.cvtColor(image, cv.COLOR_RGBA2GRAY)
        
        if isinstance(x, dict):
            x['image'] = image
        else:
            x = image
        return x


class ToContiguousArray:
    def __call__(self, x):
        if x is None:
            return None
        
        image = x['image'] if isinstance(x, dict) else x
        image = np.ascontiguousarray(image)
        
        if isinstance(x, dict):
            x['image'] = image
            x['mask'] = np.ascontiguousarray(x['mask'])
        else:
            x = image
        return x


class RandomResize:
    def __init__(self, range=(0.5, 2), interpolation=cv.INTER_LINEAR):
        self.range = range
        self.interpolation = interpolation

    def __call__(self, x):
        if x is None:
            return None
        
        log_range = np.log(self.range)
        factor_h, factor_w = np.exp(np.random.rand(2) * (log_range[1] - log_range[0]) + log_range[0])
        
        image = x['image'] if isinstance(x, dict) else x
        image = cv.resize(image, (0, 0), fx=factor_w, fy=factor_h, interpolation=self.interpolation)
    
        if isinstance(x, dict):
            x['image'] = image
            x['mask'] = cv.resize(x['mask'], (0, 0), fx=factor_w, fy=factor_h, interpolation=self.interpolation)
        else:
            x = image
        return x


class RandomResize_discrete:
    def __init__(self, values=np.concatenate([
        np.arange(0.5, 1, 0.05),
        np.arange(2, 1, -0.1)]),
            interpolation=cv.INTER_LINEAR):
        self.values = values
        self.interpolation = interpolation

    def __call__(self, x):
        if x is None:
            return None

        factor_h = np.random.choice(self.values)
        factor_w = np.random.choice(self.values)
        
        image = x['image'] if isinstance(x, dict) else x
        image = cv.resize(image, (0, 0), fx=factor_w, fy=factor_h, interpolation=self.interpolation)
        
        if isinstance(x, dict):
            x['image'] = image
            x['mask'] = cv.resize(x['mask'], (0, 0), fx=factor_w, fy=factor_h, interpolation=self.interpolation)
        else:
            x = image
        return x


class RandomScale:
    def __init__(self, range=(0.5, 2), interpolation=cv.INTER_LINEAR):
        self.range = range
        self.interpolation = interpolation

    def __call__(self, x):
        if x is None:
            return None

        log_range = np.log(self.range)
        factor = np.exp(np.random.rand() * (log_range[1] - log_range[0]) + log_range[0])
        
        image = x['image'] if isinstance(x, dict) else x
        image = cv.resize(image, (0, 0), fx=factor, fy=factor, interpolation=self.interpolation)
        
        if isinstance(x, dict):
            x['image'] = image
            x['mask'] = cv.resize(x['mask'], (0, 0), fx=factor, fy=factor, interpolation=self.interpolation)
        else:
            x = image
        return x


class RandomScale_discrete:
    def __init__(self, values=np.concatenate([
        np.arange(0.5, 1, 0.05),
        np.arange(2, 1, -0.1)]),
            interpolation=cv.INTER_LINEAR):
        self.values = values
        self.interpolation = interpolation

    def __call__(self, x):
        if x is None:
            return None

        factor = np.random.choice(self.values)
        
        image = x['image'] if isinstance(x, dict) else x
        image = cv.resize(image, (0, 0), fx=factor, fy=factor, interpolation=self.interpolation)
        
        if isinstance(x, dict):
            x['image'] = image
            x['mask'] = cv.resize(x['mask'], (0, 0), fx=factor, fy=factor, interpolation=self.interpolation)
        else:
            x = image
        return x


class RandomRotate:
    def __init__(self, range=(-45, 45), interpolation=cv.INTER_LINEAR, mode='valid'):
        self.range = range
        self.interpolation = interpolation
        self.mode = mode

    def __call__(self, x):
        if x is None:
            return None

        angle = np.random.rand() * (self.range[1] - self.range[0]) + self.range[0]

        image = x['image'] if isinstance(x, dict) else x
        if self.mode == 'valid':
            image, offset = extract_random_square(image)
            M, new_shape = rotate_valid(image, angle)
        elif self.mode == 'bound':
            M, new_shape = rotate_bound(image, angle)
        image = cv.warpAffine(image, M, new_shape, flags=self.interpolation)
        
        if isinstance(x, dict):
            x['image'] = image
            if self.mode == 'valid':
                x['mask'] = extract_square(x['mask'], offset)
                
            x['mask'] = cv.warpAffine(x['mask'], M, new_shape, flags=self.interpolation)
        else:
            x = image
        return x


class RandomRotate_discrete:
    def __init__(self, values=np.concatenate([
            np.arange(-45, 0, 5),
            np.arange(45, 0, -5)]),
            interpolation=cv.INTER_LINEAR, mode='valid'):
        self.values = values
        self.interpolation = interpolation
        self.mode = mode

    def __call__(self, x):
        if x is None:
            return None

        angle = np.random.choice(self.values)

        image = x['image'] if isinstance(x, dict) else x
        if self.mode == 'valid':
            image, offset = extract_random_square(image)
            M, new_shape = rotate_valid(image, angle)
        elif self.mode == 'bound':
            M, new_shape = rotate_bound(image, angle)
        image = cv.warpAffine(image, M, new_shape, flags=self.interpolation)
        
        if isinstance(x, dict):
            x['image'] = image
            if self.mode == 'valid':
                x['mask'] = extract_square(x['mask'], offset)
            x['mask'] = cv.warpAffine(x['mask'], M, new_shape, flags=self.interpolation)
        else:
            x = image
        return x


class RandomMedianFilter:
    def __init__(self, values=[3, 5, 7]):
        """[summary]

        Args:
            values (list, optional): 
                Random parameters. Defaults to [3, 5, 7].
        """
        self.values = values
        
    def __call__(self, x):
        if x is None:
            return None
        
        ksize = np.random.choice(self.values)
        
        image = x['image'] if isinstance(x, dict) else x
        image = cv.medianBlur(image, ksize)
        
        if isinstance(x, dict):
            x['image'] = image
        else:
            x = image
        return x


class RandomBoxFilter:
    def __init__(self, values=[3, 5, 7]):
        """Normalized BoxFilter, that is, Average Filter.

        Args:
            values (list, optional): 
                Random parameters. Defaults to [3, 5, 7].
        """
        self.values = values
        
    def __call__(self, x):
        if x is None:
            return None
        
        ksize = np.random.choice(self.values)
        ksize = (ksize, ksize)
        
        image = x['image'] if isinstance(x, dict) else x
        image = cv.boxFilter(image, ddepth=-1, ksize=ksize)
        
        if isinstance(x, dict):
            x['image'] = image
        else:
            x = image
        return x


class RandomGaussianFilter:
    def __init__(self, ksize_values=[3, 5, 7], std_values=[0]):
        """GaussianFilter. The standard deviation is calculated as ((ksize - 1) * 0.5 - 1) * 0.3 + 0.8.

        Args:
            ksize_values (list, optional): The kernel size. Defaults to [3, 5, 7].
            std_values (list, optional): 
                The standard deviation. Defaults to [0]. When set to 0, it is calculated 
                as ((ksize - 1) * 0.5 - 1) * 0.3 + 0.8.
            
        """
        self.ksize_values = ksize_values
        self.std_values = std_values
        
    def __call__(self, x):
        if x is None:
            return None
        
        ksize = np.random.choice(self.ksize_values)
        ksize = (ksize, ksize)
        std = np.random.choice(self.std_values)
        
        image = x['image'] if isinstance(x, dict) else x
        image = cv.GaussianBlur(image, ksize, sigmaX=std, sigmaY=std)
        
        if isinstance(x, dict):
            x['image'] = image
        else:
            x = image
        return x
    
    
class RandomAWGN:
    def __init__(self, range=(0.03, 0.1), func: Callable = gaussian_noise):
        self.range = range
        self.func = func
        
    def __call__(self, x):
        if x is None:
            return None
        
        std = np.random.rand() * (self.range[1] - self.range[0]) + self.range[0]
        
        image = x['image'] if isinstance(x, dict) else x
        image = self.func(image, std=std)
        
        if isinstance(x, dict):
            x['image'] = image
        else:
            x = image
        return x
    
    
class RandomAWGN_discrete:
    def __init__(self, values=np.arange(0.03, 0.11, 0.01), func: Callable = gaussian_noise):
        self.values = values
        self.func = func
        
    def __call__(self, x):
        if x is None:
            return None
        
        std = np.random.choice(self.values)
        
        image = x['image'] if isinstance(x, dict) else x
        image = self.func(image, std=std)
        
        if isinstance(x, dict):
            x['image'] = image
        else:
            x = image
        return x


class PoissonNoise:
    def __init__(self, func: Callable = poisson_noise):
        self.func = func
    
    def __call__(self, x):
        if x is None:
            return None
        
        image = x['image'] if isinstance(x, dict) else x
        image = self.func(image)
        
        if isinstance(x, dict):
            x['image'] = image
        else:
            x = image
        return x


class RandomImpulseNoise:
    def __init__(self, amount_range=(0.01, 0.05),
                 proportion_range=(0, 1)):
        """Impulse noise, also known as salt and pepper noise.

        Args:
            amount_range (tuple, optional): 
                Random range to set amount. Defaults to (0.01, 0.05).
            proportion_range (tuple, optional): 
                Random range to set salt_vs_pepper proportion. Defaults to (0, 1).
        """
        self.amount_range = amount_range
        self.proportion_range = proportion_range
        
    def __call__(self, x):
        if x is None:
            return None
        
        amount = np.random.rand() * \
            (self.amount_range[1] - self.amount_range[0]) + self.amount_range[0]
        proportion = np.random.rand() * \
            (self.proportion_range[1] - self.proportion_range[0]) + self.proportion_range[0]
        
        image = x['image'] if isinstance(x, dict) else x
        image = impulse_noise(image, amount=amount, salt_vs_pepper=proportion)
        
        if isinstance(x, dict):
            x['image'] = image
        else:
            x = image
        return x


class RandomImpulseNoise_discrete:
    def __init__(self, amount_values=[0.01, 0.02, 0.03, 0.04, 0.05],
                 proportion_values=[0, 0.25, 0.5, 0.75, 1.0]):
        """Impulse noise, also known as salt and pepper noise.

        Args:
            amount_values (list, optional): 
                Random values to set amount. Defaults to [0.01, 0.02, 0.03, 0.04, 0.05].
            proportion_values (list, optional): 
                Random values to set salt_vs_pepper proportion. Defaults to [0, 0.25, 0.5, 0.75, 1.0].
        """
        self.amount_values = amount_values
        self.proportion_values = proportion_values
        
    def __call__(self, x):
        if x is None:
            return None
        
        amount = np.random.choice(self.amount_values)
        proportion = np.random.choice(self.proportion_values)
        
        image = x['image'] if isinstance(x, dict) else x
        image = impulse_noise(image, amount=amount, salt_vs_pepper=proportion)
        
        if isinstance(x, dict):
            x['image'] = image
        else:
            x = image
        return x


# class RandomWarpAffine:
#     def __init__(self, angle_range=(-45, 45), scale_range=(0.5, 2), interpolation=cv.INTER_LINEAR):
#         self.angle_range = angle_range
#         self.scale_range = scale_range
#         self.interpolation = interpolation

#     def __call__(self, image):
#         if image is None:
#             return None

#         log_scale_range = np.log(self.scale_range)

#         angle = np.random.rand() * \
#             (self.angle_range[1] - self.angle_range[0]) + self.angle_range[0]
#         scale = np.exp(np.random.rand() * (log_scale_range[1] - log_scale_range[0]) + log_scale_range[0])

#         image, _ = extract_random_square(image)
#         M, new_shape = rotate_valid(image, angle, scale)
#         return cv.warpAffine(image, M, new_shape, flags=self.interpolation)


class Crop:
    def __init__(self, end_h, end_w, start_h=0, start_w=0):
        if not all([isinstance(index, int) for index in [start_h, end_h, start_w, end_w]]):
            raise ValueError('index must be integer!')
        self.start_h, self.end_h = start_h, end_h
        self.start_w, self.end_w = start_w, end_w

    def __call__(self, x):
        if x is None:
            return None

        image = x['image'] if isinstance(x, dict) else x
        h, w = image.shape[:2]

        # if self.start_h > h or self.start_w > w:
        #     return None
        if self.end_h > h or self.end_w > w:
            return None

        image = image[self.start_h: self.end_h,
                      self.start_w: self.end_w]
        
        if isinstance(x, dict):
            x['image'] = image
            x['mask'] = x['mask'][self.start_h: self.end_h,
                                  self.start_w: self.end_w]
        else:
            x = image
        return x


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        if x is None:
            return None

        image = x['image'] if isinstance(x, dict) else x
        h, w = image.shape[:2]
        tH, tW = self.size

        if h < tH or w < tW:
            return None

        i = np.random.randint(h - tH + 1)
        j = np.random.randint(w - tW + 1)

        image = image[i: i + tH, j: j + tW]
        
        if isinstance(x, dict):
            x['image'] = image
            x['mask'] = x['mask'][i: i + tH, j: j + tW]
        else:
            x = image
        return x


class RandomPatch:
    def __init__(self, patch_size, max_stride):
        self.patch_size = _pair(patch_size)
        self.max_stride = _pair(max_stride)

    def __call__(self, x):
        if x is None:
            return None

        image = x['image'] if isinstance(x, dict) else x
        patch_idxs, _ = calc_patch_indexs(
            image, self.patch_size, self.max_stride)

        if patch_idxs is not None:
            patch_idx = patch_idxs[np.random.randint(len(patch_idxs))]

            index_h, index_w = patch_idx
            patch_h, patch_w = self.patch_size

            image = image[index_h: index_h + patch_h, 
                          index_w: index_w + patch_w]
            
            if isinstance(x, dict):
                x['image'] = image
                x['mask'] = x['mask'][index_h: index_h + patch_h, 
                                      index_w: index_w + patch_w]
            else:
                x = image
            return x

 
class RandomNPatches:
    def __init__(self, patch_size, max_stride, N):
        self.patch_size = _pair(patch_size)
        self.max_stride = _pair(max_stride)
        self.N = N

    def __call__(self, x):
        if x is None:
            return None

        image = x['image'] if isinstance(x, dict) else x
        patch_idxs, _ = calc_patch_indexs(
            image, self.patch_size, self.max_stride)

        if patch_idxs is not None:
            if self.N < len(patch_idxs):
                patch_idxs_idxs = np.random.choice(len(patch_idxs), self.N, replace=False)
                patch_idxs = [patch_idxs[pidx_idx] for pidx_idx in patch_idxs_idxs]
            
            patch_h, patch_w = self.patch_size
            patches = []
            if isinstance(x, dict):
                patches_mask = []
            
            for index_h, index_w in patch_idxs:
                patches.append(
                    image[index_h: index_h + patch_h, 
                          index_w: index_w + patch_w])
                
                if isinstance(x, dict):
                    patches_mask.append(
                        x['mask'][index_h: index_h + patch_h, 
                                  index_w: index_w + patch_w])
            
            if isinstance(x, dict):
                x['image'] = patches
                x['mask'] = patches_mask
            else:
                x = patches
            return x


class AllPatches:
    def __init__(self, patch_size, max_stride):
        self.patch_size = _pair(patch_size)
        self.max_stride = _pair(max_stride)

    def __call__(self, x):
        if x is None:
            return None

        image = x['image'] if isinstance(x, dict) else x
        patch_idxs, _ = calc_patch_indexs(
            image, self.patch_size, self.max_stride)

        if patch_idxs is not None:
            patches = extract_image_patches(
                image, patch_idxs, self.patch_size)
            
            if isinstance(x, dict):
                x['image'] = patches
                x['mask'] = extract_image_patches(
                    x['mask'], patch_idxs, self.patch_size)
            else:
                x = patches
            return x
       
        
class SelectPatches:
    def __init__(self, area_range=(0.03, 0.6)):
        self.area_range = area_range

    def __call__(self, x):
        if x is None:
            return None
        
        if not isinstance(x, dict):
            return None
        
        area_range = self.area_range
        
        patches = x['image']
        patches_mask = x['mask']
        areas = [np.mean(patch > 0) for patch in patches_mask]
        
        new_patches = []
        new_patches_mask = []
        for idx, area in enumerate(areas):
            if area < area_range[0] or area > area_range[1]:
                continue
            
            new_patches.append(patches[idx])
            new_patches_mask.append(patches_mask[idx])
        
        if len(new_patches) == 0:
            return None
        
        x['image'] = new_patches
        x['mask'] = new_patches_mask
        return x        


class RandomTransform:
    def __init__(self, transforms, probs=None, return_label=False, debug=False):
        if probs is None:
            probs = np.ones(len(transforms))

        if len(transforms) != len(probs):
            raise ValueError('the len of transforms must be equal to that of preds!')
        if any(probs <= 0):
            raise ValueError('the prob of transforms must be positive!')

        self.transforms = transforms
        self.probs = probs
        self.return_label = return_label
        self.debug = debug

    def __call__(self, x):
        if x is None:
            return None

        rand = np.random.rand()
        if self.debug:
            print(f" rand: {rand}")
        rand *= sum(self.probs)

        for tid in range(len(self.transforms)):
            if rand <= sum(self.probs[:tid + 1]):

                if self.return_label:
                    return self.transforms[tid](x), tid
                else:
                    return self.transforms[tid](x)


class JPEGCompress:
    def __init__(self, quality=75, subsampling='4:2:0', cache_dir='./cache/'):
        """This transform can be used to save JPEG file when you call it with parameter 'fname'.
        
        quality: int
            The image quality, on a scale from 1 (worst) to 100 (best). The default is 75.
        subsampling: str or int
            Sets the subsampling for the encoder. Only 4:4:4 (0), 4:2:2 (1), 4:2:0 (2) are supported.
            The default is 4:2:0, which is the most common setting in practice.
        """
        self.quality = quality
        self.subsampling = subsampling
        self.cache_dir = cache_dir
        keep_dir_valid(cache_dir)

    def __call__(self, x, fname=None):
        if x is None:
            return None

        image = x['image'] if isinstance(x, dict) else x
        im = Image.fromarray(image)
        
        fname = f"temp_p{getpid()}.jpg" if fname is None else fname
        fpath = path_join(self.cache_dir, fname)
        # quality of type numpy.int32 is invalid
        # the filename f"temp_p{getpid()}.jpg" is used for the compatibility of multiprocess
        im.save(fpath, quality=int(self.quality), subsampling=self.subsampling)
        image = np.array(Image.open(fpath))

        if isinstance(x, dict):
            x['image'] = image
        else:
            x = image
            
        return x


class RandomJPEGCompress:
    def __init__(self, quality_values, quality_probs=None,
                 subsampling_probs=[1, 1, 1], cache_dir='./cache/'):
        """This transform can be used to save JPEG file when you call it with parameter 'fname'.
        
        subsampling_probs: list or array
            Probability of ['4:4:4', '4:2:2', '4:2:0'], respectively.
        """
        self.quality_values = quality_values
        if quality_probs is None:
            quality_probs = np.ones_like(self.quality_values)

        self.quality_probs = np.divide(quality_probs, sum(quality_probs))
        if sum(self.quality_probs) != 1:
            self.quality_probs[self.quality_probs.argmax()] += 1 - \
                sum(self.quality_probs)

        self.subsampling_probs = np.divide(
            subsampling_probs, sum(subsampling_probs))
        if sum(self.subsampling_probs) != 1:
            self.subsampling_probs[self.subsampling_probs.argmax(
            )] += 1 - sum(self.subsampling_probs)

        self.cache_dir = cache_dir
        keep_dir_valid(cache_dir)

    def __call__(self, x, fname=None):
        if x is None:
            return None

        image = x['image'] if isinstance(x, dict) else x
        im = Image.fromarray(image)
        quality = np.random.choice(self.quality_values, p=self.quality_probs)
        subsampling = np.random.choice([0, 1, 2], p=self.subsampling_probs)

        fname = f"temp_p{getpid()}.jpg" if fname is None else fname
        fpath = path_join(self.cache_dir, fname)
        im.save(fpath, quality=int(quality), subsampling=subsampling)
        image = np.array(Image.open(fpath))

        if isinstance(x, dict):
            x['image'] = image
        else:
            x = image
            
        return x


class WEBPCompress:
    def __init__(self, quality=75, cache_dir='./cache/'):
        """This transform can be used to save JPEG file when you call it with parameter 'fname'.
        
        quality: int
            The image quality, on a scale from 1 (worst) to 100 (best). The default is 75.
        """
        self.quality = quality
        self.cache_dir = cache_dir
        keep_dir_valid(cache_dir)

    def __call__(self, x, fname=None):
        if x is None:
            return None

        image = x['image'] if isinstance(x, dict) else x
        im = Image.fromarray(image)

        fname = f"temp_p{getpid()}.webp" if fname is None else fname
        fpath = path_join(self.cache_dir, fname)
        im.save(fpath, quality=int(self.quality))
        image = np.array(Image.open(fpath))

        if isinstance(x, dict):
            x['image'] = image
        else:
            x = image
            
        return x


class RandomWEBPCompress:
    def __init__(self, quality_values, quality_probs=None, cache_dir='./cache/'):
        """This transform can be used to save JPEG file when you call it with parameter 'fname'.
        """
        self.quality_values = quality_values
        if quality_probs is None:
            quality_probs = np.ones_like(self.quality_values)
        self.quality_probs = np.divide(quality_probs, sum(quality_probs))
        if sum(self.quality_probs) != 1:
            self.quality_probs[self.quality_probs.argmax()] += 1 - \
                self.quality_probs

        self.cache_dir = cache_dir
        keep_dir_valid(cache_dir)

    def __call__(self, x, fname=None):
        if x is None:
            return None

        image = x['image'] if isinstance(x, dict) else x
        im = Image.fromarray(image)
        quality = np.random.choice(self.quality_values, p=self.quality_probs)

        fname = f"temp_p{getpid()}.webp" if fname is None else fname
        fpath = path_join(self.cache_dir, fname)
        im.save(fpath, quality=int(quality))
        im = Image.open(fpath)

        if isinstance(x, dict):
            x['image'] = image
        else:
            x = image
            
        return x