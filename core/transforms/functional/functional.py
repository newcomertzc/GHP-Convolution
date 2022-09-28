import math
import numpy as np
import cv2 as cv

from ...common_types import *


__all__ = [
    'rotate_bound', 'rotate_valid', 'impulse_noise', 'poisson_noise', 'poisson_noise_v2', 
    'extract_square', 'extract_random_square', 
    'calc_patch_indexs', 'extract_image_patches', 'rebuild_image_patches', 
]


def rotate_bound(image: ndarray, angle: float, scale: float = 1.0) -> tuple:
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix, then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((cX, cY), angle, scale)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return M, (nW, nH)


def rotate_valid(image: ndarray, angle: float, scale: float = 1.0) -> tuple:
    # grab the dimensions of a squared image and then determine the center
    # it gives an output which doesn't contain padding
    # you should use extract_random_square before use rotate_valid
    (h, w) = image.shape[:2]
    assert h == w, 'rotate_valid only supports an image with the same height and width!'
    c = w // 2

    # grab the rotation matrix, then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((c, c), angle, scale)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    nW = int(w / (cos + sin) * scale**2)

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - c
    M[1, 2] += (nW / 2) - c

    return M, (nW, nW)


def impulse_noise(image: ndarray, amount: float = 0.05, salt_vs_pepper: float = 0.5) -> ndarray:
    temp = image.copy().ravel()
    
    idxs = np.random.choice(
        len(temp), round(len(temp) * amount), replace=False)
    idxs_sidxs = np.random.choice(
        len(idxs), round(len(idxs) * salt_vs_pepper), replace=False)
    
    idxs_salt = idxs[idxs_sidxs]
    idxs_pepper = np.delete(idxs, idxs_sidxs)
    
    temp[idxs_salt] = 255 if image.dtype == np.uint8 else 1.0
    temp[idxs_pepper] = 0
    return temp.reshape(image.shape)


def poisson_noise(image: ndarray, vals: Optional[int] = None) -> ndarray:
    if vals is None:
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
    
    temp = image / 255.0 if image.dtype == np.uint8 else image
    temp = np.random.poisson(temp * vals) / vals
    temp[temp > 1.0] = 1.0
    temp[temp < 0.0] = 0.0
    
    if image.dtype == np.uint8:
        temp = np.array(temp * 255, dtype=np.uint8)
    return temp


def poisson_noise_v2(image: ndarray, vals: Optional[int] = None) -> ndarray:
    if vals is None:
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
    
    temp = np.array(image / 255.0, dtype=np.float32) if image.dtype == np.uint8 else image
    temp = np.random.poisson(temp * vals) / vals
    temp[temp > 1.0] = 1.0
    temp[temp < 0.0] = 0.0
    
    if image.dtype == np.uint8:
        temp = np.array(temp * 255, dtype=np.uint8)
    return temp


def extract_square(image: ndarray, offset: int) -> ndarray:
    # extract square part from image
    (h, w) = image.shape[:2]
    if h == w:
        return image
    
    offset = sum(offset)

    if h > w:
        return image[offset: offset + w, :]
    if h < w:
        return image[:, offset: offset + h]


def extract_random_square(image: ndarray) -> ndarray:
    # extract random square part from image
    (h, w) = image.shape[:2]
    if h == w:
        return image, (0, 0)

    diff = abs(h - w)
    offset = np.random.randint(diff + 1)

    if h > w:
        return image[offset: offset + w, :], (offset, 0)
    if h < w:
        return image[:, offset: offset + h], (0, offset)


def calc_patch_indexs(image: ndarray, patch_size: tuple = (224, 224), 
                      max_stride: tuple = (160, 160)) -> tuple:
    """Calculate the indexs of each patch using a adaptive stride. 
    """
    patch_indexs = []

    image_h, image_w = image.shape[:2]
    patch_h, patch_w = patch_size
    max_stride_h, max_stride_w = max_stride

    if image_h < patch_h or image_w < patch_w:
        return None, None

    patch_nh, patch_nw = (image_h - patch_h) / max_stride_h + \
        1, (image_w - patch_w) / max_stride_w + 1
    patch_nh, patch_nw = math.ceil(patch_nh), math.ceil(patch_nw)

    if patch_nh == 1:
        stride_h = 0
    else:
        stride_h = math.ceil((image_h - patch_h) / (patch_nh - 1))

    if patch_nw == 1:
        stride_w = 0
    else:
        stride_w = math.ceil((image_w - patch_w) / (patch_nw - 1))

    for nh in range(patch_nh - 1):
        for nw in range(patch_nw - 1):
            patch_indexs.append((nh*stride_h, nw*stride_w))
        patch_indexs.append((nh*stride_h, image_w - patch_w))

    for nw in range(patch_nw - 1):
        patch_indexs.append((image_h - patch_h, nw*stride_w))
    patch_indexs.append((image_h - patch_h, image_w - patch_w))

    return patch_indexs, (stride_h, stride_w)


def extract_image_patches(image: ndarray, patch_indexs: List[tuple], 
                          patch_size: tuple = (224, 224)) -> ndarray:
    """Cut image into patches of specific shape using a adaptive stride. 
    """
    patches = []
    patch_h, patch_w = patch_size

    for index_h, index_w in patch_indexs:
        patch = image[
            index_h: index_h + patch_h,
            index_w: index_w + patch_w]
        patches.append(patch)

    return np.array(patches)


def rebuild_image_patches(patches: List[ndarray], patch_indexs: List[tuple], 
                          image_size: tuple, dtype: type = np.uint8) -> tuple:
    """Rebuild image from cut patches.
    """
    image = np.zeros(shape=image_size)
    weight = np.zeros(shape=image_size)

    patch_h, patch_w = patches[0].shape[:2]
    for i, (index_h, index_w) in enumerate(patch_indexs):
        image[
            index_h: index_h + patch_h,
            index_w: index_w + patch_w] += patches[i]
        weight[
            index_h: index_h + patch_h,
            index_w: index_w + patch_w] += 1

    return np.array(image / weight, dtype=dtype), weight