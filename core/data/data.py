import cv2 as cv
import numpy as np

from PIL import Image
from glob import glob
from torchvision.datasets import ImageFolder as ClfDataset

from ..utils import *
from ..common_types import *
from ..transforms import Remain


__all__ = [
    'Nothing', 'NotNone', 'NotSmallerthanSize', 'PILLoader', 'OpenCVLoader', 
    'ImageNet', 'ClfDataset', 'SegDataset', 'TestDataset'
]


class Nothing:
    def __call__(self, x: Any) -> bool:
        return True


class NotNone:
    def __call__(self, x: Any) -> bool:
        return x is not None


class NotSmallerthanSize:
    def __init__(self, size: tuple) -> None:
        self.size = size
    
    def __call__(self, x: _valid_data_types) -> bool:
        if x is None:
            return False
        
        if isinstance(x, dict):
            x = x['image']
            
        return x.shape[0] >= self.size[0] and x.shape[1] >= self.size[1]

        
class PILLoader:
    def __init__(self, mode: str = None) -> None:
        self.mode = mode
    
    def __call__(self, path: str) -> ImageFile:
        im = Image.open(path)
        
        if self.mode is not None:
            im = im.convert(self.mode)
        return im


class OpenCVLoader:
    def __init__(self, color: int = cv.COLOR_BGR2RGB) -> None:
        self.color = color
    
    def __call__(self, path: str) -> ndarray:
        imarr = cv.imread(path)
        
        if self.color is not None:
            imarr = cv.cvtColor(imarr, self.color)
        return imarr


class ImageNet:
    def __init__(
        self, 
        train_dirs: List[str], 
        val_dir: str, 
        cat_mapping: Optional[str] = None
    ) -> None:
        self.train_dirs = sorted(train_dirs)
        self.val_dir = val_dir

        if cat_mapping:
            self.cat_mapping = {}
            with open(cat_mapping, 'r') as f:
                content = f.read()
            content = content.split('\n')[:-1]
            for c in content:
                self.cat_mapping[c[:9]] = c[10:]
        else:
            self.cat_mapping = None

    def load_train_images(
        self, 
        cats: Union[str, List[str]], 
        image_num: int = 100, 
        return_path: bool = True
    ) -> Union[List[str], List[ndarray]]:
        if isinstance(cats, str):
            cats = [cats]

        images = []
        for cat in cats:
            cat_id = get_base_name(cat)
            image_paths = sorted(glob(path_join(cat, '*')))
            
            image_num_real = image_num if image_num <= len(image_paths) else len(image_paths)
            if self.cat_mapping:
                print(
                    f"loading { image_num_real } images of { cat_id } - { self.cat_mapping[cat_id] }...")
            else:
                print(f"loading { image_num_real } images of { cat_id }...")
            
            if image_num_real != len(image_paths):
                image_indexs = np.random.choice(
                    len(image_paths), image_num_real, replace=False)
            else:
                image_indexs = np.arange(len(image_paths))

            for index in image_indexs:
                path = image_paths[index]
                if return_path:
                    images.append(path)
                else:
                    images.append(np.array(Image.open(path)))
        return images

    def load_val_images(self, image_num: int = 1000, return_path: bool = True) -> Union[List[str], List[ndarray]]:
        print(f"loading { image_num } images of val...")

        image_paths = sorted(glob(path_join(self.val_dir, '*')))
        rand_indexs = np.random.choice(
            len(image_paths), image_num, replace=False)

        if return_path:
            return [image_paths[index] for index in rand_indexs]
        else:
            return [np.array(Image.open(image_paths[index])) for index in rand_indexs]


class SegDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        transform: Callable = Remain(), 
        transform_mask: Callable = Remain(), 
        loader: Callable = PILLoader(), 
        pack_data_before_transform: bool = False
    ) -> None:
        """Segmentation Dataset.
        
        For an image file named A.png, its mask is expected to be named A_mask.png. 
        """
        self.images = sorted(glob(path_join(root_dir, 'image', '*')))
        self.masks = sorted(glob(path_join(root_dir, 'mask', '*')))
        if len(self.images) != len(self.masks):
            raise ValueError('length of images and masks must be equal!')
        self.loader = loader
        self.pack_data_before_transform = pack_data_before_transform
        self.transform = transform
        self.transform_mask = transform_mask
        
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index: int) -> Optional[dict]:
        image = self.loader(self.images[index])
        mask = self.loader(self.masks[index])
        mask = mask.convert('L')
        
        if self.pack_data_before_transform:
            x = {'image': image, 'mask': mask}
            x = self.transform(x)
        else:
            image = self.transform(image)
            mask = self.transform_mask(mask)
                
            x = {'image': image, 'mask': mask}
        
        return x


class TestDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        transform: Callable = Remain(), 
        loader: Callable = PILLoader()
    ) -> None:
        """Test Dataset. 
        """
        self.images = sorted(glob(path_join(root_dir, '*')))
        self.loader = loader
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index: int) -> dict:
        image = self.loader(self.images[index])
        file_name = get_base_name(self.images[index])
        image_size = image.size[1::-1]
        
        image = self.transform(image)
        x = {'image': image, 'file_name': file_name, 'image_size': image_size}
        
        return x