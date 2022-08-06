import torch
import random
import numpy as np

from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO

from ..data import *
from ...utils import *
from ...common_types import *
from ...transforms import ToRGB, Remain


__all__ = [
    'collate_fn_seg', 'collate_fn_test', 'get_clf_dataloader', 'get_clf_dataset_file', 'get_seg_dataloader',
    'get_seg_dataloader_coco', 'get_seg_dataset_file_coco',
]
    
    
def collate_fn_seg(data: List[dict]) -> dict:
    images = []
    masks = []
    for row in data:
        images.append(row['image'])
        masks.append(row['mask'])
        
    new_data = {}
    new_data['image'] = torch.stack(images)
    new_data['mask'] = torch.LongTensor(masks)
    return new_data


def collate_fn_test(data: List[dict]) -> dict: 
    images = []
    file_names = []
    for row in data:
        images.append(row['image'])
        file_names.append(row['file_name'])
        
    new_data = {}
    new_data['image'] = torch.stack(images)
    new_data['file_name'] = file_names
    return new_data


def get_clf_dataloader(
    paths: List[str],
    transform: Callable, 
    after_transform: Callable = Remain(), 
    batch_size: int = 32, 
    shuffle: bool = False, 
    drop_last: bool = False, 
    single_patch: bool = False, 
    condition: Callable = NotNone()
) -> Generator:
    """Get a classification dataloader.
    """
    paths = paths.copy()
    if shuffle:
        random.shuffle(paths)

    batch_image = []
    batch_label = []
    for path in paths:
        image = np.array(Image.open(path))
        patches, label = transform(image)

        if condition(patches):
            if not single_patch:
                image = [after_transform(patch) for patch in patches]
                label = [label] * len(image)

                batch_image.extend(image)
                batch_label.extend(label)
            else:
                image = after_transform(patches)

                batch_image.append(image)
                batch_label.append(label)
        del image, patches, label
                
        while len(batch_image) > batch_size:
            batch_image_out = torch.stack(batch_image[:batch_size])
            batch_label_out = torch.LongTensor(batch_label[:batch_size])

            yield batch_image_out, batch_label_out
            del batch_image_out, batch_label_out
            
            batch_image = batch_image[batch_size:]
            batch_label = batch_label[batch_size:]

    if len(batch_image) == batch_size or (not drop_last and len(batch_image) > 0):
        batch_image = torch.stack(batch_image)
        batch_label = torch.LongTensor(batch_label)

        yield batch_image, batch_label
        

def get_clf_dataset_file(
    paths: List[str], 
    transform: Callable, 
    label_names: dict, 
    save_dir: str = 'clf_dataset/', 
    condition: Callable = NotNone()
) -> None:
    """Get a classification dataset saved as file.
    """
    class_names = []
    for label, name in label_names.items():
        class_name = f"{label:03d}_{name}"
        class_names.append(class_name)
        
        class_dir = path_join(save_dir, class_name)
        keep_dir_valid(class_dir)

    imgId = 1
    for path in tqdm(paths):
        image = np.array(Image.open(path))
        patch, label = transform(image)

        if condition(patch):
            im = Image.fromarray(patch)
            
            fname = get_base_name(fname)
            fname = (f"{imgId:07d}_{fname.replace('_', '-')}_"
                     f"{label_names[label]}.png")
            
            im.save(path_join(save_dir, class_names[label], fname))
            imgId += 1
            
            del im
        del image, patch, label


def get_seg_dataloader(
    dataset: Dataset, 
    after_transform: Callable, 
    batch_size: int = 32, 
    shuffle: bool = False, 
    drop_last: bool = False, 
    single_patch: bool = False, 
    condition: Callable = NotNone()
) -> Generator:
    """Get a segmentation dataloader.
    """
    data_indexs = list(np.arange(len(dataset)))
    if shuffle:
        random.shuffle(data_indexs)

    batch_image = []
    batch_label = []
    for data_index in data_indexs:
        x = dataset[data_index]
        
        if not condition(x):
            continue
        
        if not single_patch:
            image = [after_transform(patch) for patch in x['image']]
            mask = x['mask']

            batch_image.extend(image)
            batch_label.extend(mask)
        else:
            image = after_transform(x['image'])
            mask = x['mask']

            batch_image.append(image)
            batch_label.append(mask)
        del x, image, mask
            
        while len(batch_image) > batch_size:
            batch_image_out = torch.stack(batch_image[:batch_size])
            batch_label_out = torch.LongTensor(batch_label[:batch_size])

            yield {'image': batch_image_out, 'mask': batch_label_out}
            del batch_image_out, batch_label_out
            
            batch_image = batch_image[batch_size:]
            batch_label = batch_label[batch_size:]

    if len(batch_image) == batch_size or (not drop_last and len(batch_image) > 0):
        batch_image = torch.stack(batch_image)
        batch_label = torch.LongTensor(batch_label)

        yield {'image': batch_image, 'mask': batch_label}



def get_seg_dataloader_coco(
    coco: COCO, 
    image_dir: str, 
    pre_transform: Callable, 
    transform: Callable, 
    after_transform: Callable, 
    batch_size: int = 32, 
    area_range: tuple = (0.03, 0.6),
    shuffle: bool = False, 
    drop_last: bool = False, 
    single_patch: bool = False,
    condition: Callable = NotNone()
) -> Generator:
    """Get a segmentation dataloader based on COCO.
    """
    imgIds = sorted(coco.getImgIds())
    toRGB = ToRGB()  
    
    if shuffle:
        random.shuffle(imgIds)
    
    batch_image = []
    batch_label = []
    for imgId in imgIds:
        annIds = coco.getAnnIds(imgId, iscrowd=False)
        if len(annIds) == 0:
            continue
        
        # parameter of type numpy.int32 is invalid to pycocotools
        annId = int(np.random.choice(annIds))
        ann = coco.loadAnns(annId)[0]
        
        img = coco.loadImgs(imgId)[0]
        imarr = np.array(Image.open(path_join(image_dir, img['file_name'])))
        mask = coco.annToMask(ann)
        
        # Plan A
        # -----
        # crop image and mask with bbox
        w0, h0, w, h = ann['bbox']
        w1, h1 = int(np.ceil(w0 + w)), int(np.ceil(h0 + h))
        w0, h0 = int(w0), int(h0)
        
        # # Plan B
        # # -----
        # # crop image and mask with bbox
        # w0, h0, w, h = ann['bbox']
        # w1, h1 = int(w0 + w), int(h0 + h)
        # w0, h0 = int(np.ceil(w0)), int(np.ceil(h0))
        
        imarr = imarr[h0: h1, w0: w1]
        mask = mask[h0: h1, w0: w1]
        
        x = {'image': imarr, 'mask': mask}
        x, label = pre_transform(x)
        label += 1  # background is class 0
        imarr, mask = x['image'], x['mask']
        
        # random choose an image as background
        bgImdId = int(np.random.choice(imgIds))
        
        bgImg = coco.loadImgs(bgImdId)[0]
        bgImarr = np.array(Image.open(path_join(image_dir, bgImg['file_name'])))
        
        area = ann['area'] * (imarr.shape[0] / (h1 - h0)) * (imarr.shape[1] / (w1 - w0))
        areaBg = bgImarr.shape[0] * bgImarr.shape[1]
        
        # area constraint
        if area < area_range[0] * areaBg or area > area_range[1] * areaBg:
            continue
        
        # size constraint
        target_size = bgImarr.shape[:2]
        imarr, offset = random_zero_pad(imarr, target_size)
        if imarr is not None:
            mask = zero_pad(mask, target_size, offset)
        else:
            continue
        
        # convert rgba and gray images to rgb images
        imarr, bgImarr = toRGB(imarr), toRGB(bgImarr)
        image = mask[:, :, np.newaxis] * imarr + (1 - mask[:, :, np.newaxis]) * bgImarr
        mask = mask * label
        
        x = {'image': image, 'mask': mask}
        x = transform(x)
        
        if not condition(x):
            continue
        
        if not single_patch:
            image = [after_transform(patch) for patch in x['image']]
            mask = x['mask']

            batch_image.extend(image)
            batch_label.extend(mask)
        else:
            image = after_transform(x['image'])
            mask = x['mask']

            batch_image.append(image)
            batch_label.append(mask)
        del x, imarr, bgImarr, image, mask
            
        while len(batch_image) > batch_size:
            batch_image_out = torch.stack(batch_image[:batch_size])
            batch_label_out = torch.LongTensor(batch_label[:batch_size])

            yield {'image': batch_image_out, 'mask': batch_label_out}
            del batch_image_out, batch_label_out
            
            batch_image = batch_image[batch_size:]
            batch_label = batch_label[batch_size:]

    if len(batch_image) == batch_size or (not drop_last and len(batch_image) > 0):
        batch_image = torch.stack(batch_image)
        batch_label = torch.LongTensor(batch_label)

        yield {'image': batch_image, 'mask': batch_label}        
        

def get_seg_dataset_file_coco(
    coco: COCO,
    image_dir: str, 
    pre_transform: Callable, 
    transform: Callable, 
    label_names: List[str], 
    save_dir: str = 'seg_dataset_coco/',
    area_range: tuple = (0.03, 0.6), 
    single_patch: bool = False, 
    condition: Callable = NotNone()
) -> None:
    """Get a segmentation dataset saved as file based on COCO.
    """
    imgIds = sorted(coco.getImgIds())
    toRGB = ToRGB()
    
    keep_dir_valid(path_join(save_dir, 'image'))
    keep_dir_valid(path_join(save_dir, 'mask'))
    
    for imgId in tqdm(imgIds):
        annIds = coco.getAnnIds(imgId, iscrowd=False)
        if len(annIds) == 0:
            continue
        
        # parameter of type numpy.int32 is invalid to pycocotools
        annId = int(np.random.choice(annIds))
        ann = coco.loadAnns(annId)[0]
        
        img = coco.loadImgs(imgId)[0]
        imarr = np.array(Image.open(path_join(image_dir, img['file_name'])))
        mask = coco.annToMask(ann)
        
        # Plan A
        # -----
        # crop image and mask with bbox
        w0, h0, w, h = ann['bbox']
        w1, h1 = int(np.ceil(w0 + w)), int(np.ceil(h0 + h))
        w0, h0 = int(w0), int(h0)
        
        # # Plan B
        # # -----
        # # crop image and mask with bbox
        # w0, h0, w, h = ann['bbox']
        # w1, h1 = int(w0 + w), int(h0 + h)
        # w0, h0 = int(np.ceil(w0)), int(np.ceil(h0))
        
        imarr = imarr[h0: h1, w0: w1]
        mask = mask[h0: h1, w0: w1]
        
        x = {'image': imarr, 'mask': mask}
        x, label = pre_transform(x)
        label += 1  # background is class 0
        imarr, mask = x['image'], x['mask']
        
        # random choose an image as background
        bgImgId = int(np.random.choice(imgIds))
        bgImg = coco.loadImgs(bgImgId)[0]
        bgImarr = np.array(Image.open(path_join(image_dir, bgImg['file_name'])))
        
        area = ann['area'] * (imarr.shape[0] / (h1 - h0)) * (imarr.shape[1] / (w1 - w0))
        areaBg = bgImarr.shape[0] * bgImarr.shape[1]
        
        # area constraint
        if area < area_range[0] * areaBg or area > area_range[1] * areaBg:
            continue
        
        # size constraint
        target_size = bgImarr.shape[:2]
        imarr, offset = random_zero_pad(imarr, target_size)
        if imarr is not None:
            mask = zero_pad(mask, target_size, offset)
        else:
            continue
        
        # convert rgba and gray images to rgb images
        imarr, bgImarr = toRGB(imarr), toRGB(bgImarr)
        image = mask[:, :, np.newaxis] * imarr + (1 - mask[:, :, np.newaxis]) * bgImarr
        mask = mask * label
        
        x = {'image': image, 'mask': mask}
        x = transform(x)
        
        if not condition(x):
            continue
        
        if not single_patch:
            if len(x['image']) == 1:
                image, mask = x['image'][0], x['mask'][0]
                
                fname = f"ann{annId:012d}_img{bgImgId:012d}_{label_names[label]}.png"
                fname_mask = f"ann{annId:012d}_img{bgImgId:012d}_{label_names[label]}_mask.png"
                
                Image.fromarray(image).save(path_join(save_dir, 'image', fname))
                Image.fromarray(mask).save(path_join(save_dir, 'mask', fname_mask))
            else:
                for idx, image in enumerate(x['image']):
                    mask = x['mask'][idx]
                    fname = f"ann{annId:012d}_img{bgImgId:012d}_{label_names[label]}_patch{idx + 1:02d}.png"
                    fname_mask = f"ann{annId:012d}_img{bgImgId:012d}_{label_names[label]}_patch{idx + 1:02d}_mask.png"
                    
                    Image.fromarray(image).save(path_join(save_dir, 'image', fname))
                    Image.fromarray(mask).save(path_join(save_dir, 'mask', fname_mask))
        else:
            image, mask = x['image'], x['mask']            
            fname = f"ann{annId:012d}_img{bgImgId:012d}_{label_names[label]}.png"
            fname_mask = f"ann{annId:012d}_img{bgImgId:012d}_{label_names[label]}_mask.png"
            
            Image.fromarray(image).save(path_join(save_dir, 'image', fname))
            Image.fromarray(mask).save(path_join(save_dir, 'mask', fname_mask))
            
        del x, imarr, bgImarr, image, mask