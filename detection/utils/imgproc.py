"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import numpy as np
from skimage import io
import cv2
from typing import Tuple, Union
import torch
import torch.nn.functional as F

def loadImage(img_file: str) -> np.ndarray:
    """Given the image path, reads the image into array
    
    Args:
        img_file: Image file path

    Returns:
        img: RGB image array of shape (Height, Width, 3)
    """
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    return img

def normalizeMeanVariance(in_img: Union[np.ndarray, torch.Tensor], 
                          mean: Tuple[float,float,float] = (0.485, 0.456, 0.406), 
                          variance: Tuple[float, float, float] = (0.229, 0.224, 0.225)
                          ) -> Union[np.ndarray, torch.Tensor]:
    """Given a torch tensor or numpy array normalises them using predefined mean and variance
    
    Args:
        in_image: Numpy array or torch tensor containing input image. 
                   Shape is (Batch_size, Num_channels, Height, Width)
        mean: Tuple containing the mean value for each channel. Should be of length Num_channels
        variance: Tuple containing the variance value for each channel. Should be of length Num_channels

    Returns:
        in_img: Normalised Numpy array or torch tensor of shape (Batch_size, Num_channels, Height, Width)
    """
    if isinstance(in_img, torch.Tensor):
        if in_img.dtype != torch.float32: in_img = in_img.to(dtype=torch.float32)
        in_img.div_(255)
        in_img.add_(torch.tensor([-mean[0], -mean[1], -mean[2]], dtype=torch.float32, 
                    device=in_img.device).view(1, -1, 1, 1))
        in_img.div_(torch.tensor([variance[0], variance[1], variance[2]], dtype=torch.float32, 
                    device=in_img.device).view(1, -1, 1, 1))
    else:
        if in_img.dtype != np.float32: in_img = in_img.astype(np.float32)
        in_img /= 255
        in_img -= np.array([mean[0], mean[1], mean[2]], dtype=np.float32)
        in_img /= np.array([variance[0], variance[1], variance[2]], dtype=np.float32)
    return in_img

def resize_aspect_ratio(img: np.ndarray, square_size: int, interpolation: str, 
                        mag_ratio: float = 1.0, device="cpu"
                        ) -> Tuple[torch.Tensor, float, Tuple[int, int]]:
    """Resizes the given input image according to given parameters and maintaining aspect ratio
    
    Args:
        img: Input image array to be resized. Shape is (Height, Width, 3)
        square_size: Maximum height or width of the output image
        interpolation: Interpolation type for resizing
        mag_ratio: Magnify the input image by this scale
        device: Device to place the resized image in

    Returns:
        resized: Image resized according to input parameters. Shape is (1, Num_channels, Height, Width)
        ratio: ratio between output image scale and input image scale
        size_heatmap: Tuple containing output heatmap size
    """
    height, width, channel = img.shape
    # magnify image size
    target_size = mag_ratio * max(height, width)
    # set original image size
    if target_size > square_size:
        target_size = square_size 
    ratio = target_size / max(height, width)    
    target_h, target_w = int(height * ratio), int(width * ratio)
    img = torch.from_numpy(np.expand_dims(img, 
                    0)).to(device=device).permute(0, 3, 1, 2).to(dtype=torch.float32).contiguous()
    proc = F.interpolate(img, (target_h, target_w), mode=interpolation, align_corners=False)
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = torch.zeros(1, channel, target_h32, target_w32, dtype=torch.float32, device=device)
    resized[:, :, 0:target_h, 0:target_w] = proc
    target_h, target_w = target_h32, target_w32
    size_heatmap = (int(target_w/2), int(target_h/2))
    return resized, ratio, size_heatmap
