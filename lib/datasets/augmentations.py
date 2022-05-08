import cv2
import torch
import random
import numpy as np
from copy import deepcopy

cv2.setNumThreads(0) 

import PIL
from PIL import ImageEnhance, ImageFilter

from torchvision.datasets import ImageFolder
import torch.nn.functional as F


def to_pil(im):
    if isinstance(im, PIL.Image.Image):
        return im
    elif isinstance(im, torch.Tensor):
        return PIL.Image.fromarray(np.asarray(im))
    elif isinstance(im, np.ndarray):
        return PIL.Image.fromarray(im)
    else:
        raise ValueError('Type not supported', type(im))


def to_torch_uint8(im):
    if isinstance(im, PIL.Image.Image):
        im = torch.as_tensor(np.asarray(im).astype(np.uint8))
    elif isinstance(im, torch.Tensor):
        assert im.dtype == torch.uint8
    elif isinstance(im, np.ndarray):
        assert im.dtype == np.uint8
        im = torch.as_tensor(im)
    else:
        raise ValueError('Type not supported', type(im))
    if im.dim() == 3:
        assert im.shape[-1] in {1, 3}
    return im

class NpScaleAndRotate:
    # bboxes assumed to be ROIs [x1 y1 x2 y2] (size [N 4] for N objects)
    def __call__(self, im, depth=None, bboxes=None, K=None, output_size=None):
        # Check image
        assert isinstance(im, np.ndarray)
        assert im.dtype == np.uint8
        assert len(im.shape) >= 2
        if output_size is None:
            output_size = im.shape[:2]
        height, width = output_size
        s = np.random.uniform(1.0, 1.5) # random scaling
        # We can only do a small angle rotation or close to 180 because
        # of the bounding boxes, which would get messed up.
        angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
        if np.random.uniform(0.0, 1.0) < 0.5: # random 180 degree rotation
            angle += 180
        # Rotating and scale K matrix. Note that this will add off-diagonals!!!
        T = np.eye(3, dtype=np.float32)
        T[:2,:] = cv2.getRotationMatrix2D((width/2.-.5, height/2.-.5), angle, s)
        if K is not None:
            # Correct the camera matrix.
            # Fix the camera matrix for the initial resize
            if im.shape != output_size:
                S = np.eye(3)
                S[0,0] = width/im.shape[1]
                S[1,1] = height/im.shape[0]
                K = S @ K
            # K may have off diagonals in the 2x2 block now
            K = T @ K
        T = T[:2,:]
        cv_size = output_size[::-1] # OpenCV uses [width height] not [height width]
        im = cv2.warpAffine(im, T, cv_size, cv2.INTER_LINEAR)
        if depth is not None:
            depth = cv2.warpAffine(depth, T, cv_size, cv2.INTER_NEAREST)
        if bboxes is not None:
            bboxes = (bboxes.reshape(-1,2) @ T[:2,:2].T + T[None,:2,2]).reshape(-1,4)
        return im, depth, bboxes, K

class PillowBlur:
    def __init__(self, p=0.4, factor_interval=(1, 3)):
        self.p = p
        self.factor_interval = factor_interval

    def __call__(self, im, depth=None, bboxes=None, K=None, output_size=None):
        im = to_pil(im)
        k = random.randint(*self.factor_interval)
        im = im.filter(ImageFilter.GaussianBlur(k))
        return im, depth, bboxes, K


class PillowRGBAugmentation:
    def __init__(self, pillow_fn, p, factor_interval):
        self._pillow_fn = pillow_fn
        self.p = p
        self.factor_interval = factor_interval

    def __call__(self, im, depth=None, bboxes=None, K=None, output_size=None):
        im = to_pil(im)
        if random.random() <= self.p:
            im = self._pillow_fn(im).enhance(factor=random.uniform(*self.factor_interval))
        return im, depth, bboxes, K


class PillowSharpness(PillowRGBAugmentation):
    def __init__(self, p=0.3, factor_interval=(0., 50.)):
        super().__init__(pillow_fn=ImageEnhance.Sharpness,
                         p=p,
                         factor_interval=factor_interval)


class PillowContrast(PillowRGBAugmentation):
    def __init__(self, p=0.3, factor_interval=(0.2, 50.)):
        super().__init__(pillow_fn=ImageEnhance.Contrast,
                         p=p,
                         factor_interval=factor_interval)


class PillowBrightness(PillowRGBAugmentation):
    def __init__(self, p=0.5, factor_interval=(0.1, 6.0)):
        super().__init__(pillow_fn=ImageEnhance.Brightness,
                         p=p,
                         factor_interval=factor_interval)


class PillowColor(PillowRGBAugmentation):
    def __init__(self, p=0.3, factor_interval=(0.0, 20.0)):
        super().__init__(pillow_fn=ImageEnhance.Color,
                         p=p,
                         factor_interval=factor_interval)

