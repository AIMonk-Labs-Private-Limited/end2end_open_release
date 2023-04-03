# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import time
from typing import List, Dict, Tuple,  Any
from collections import OrderedDict

def time_synchronized() -> float:
    """Returns the current time after synchronising all the cuda kernels

    Returns:
        Current timestamp
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def clean_checkpoint(ckpt: 'OrderedDict[str, torch.Tensor]',
                     model: nn.Module) -> Dict[str, torch.Tensor]:
    """Cleans the name of parameters containing distributed training headers

    Args:
        ckpt: Dictionary containing parameter names mapped to torch Tensor
        model: Model object

    Returns:
        A new dictionary with cleaned names mapped to corresponding parameters
    """
    new_ckpt = {}
    current_state_dict = model.state_dict()
    for i,k in ckpt.items():
        if i[0:6] == "module":
            name = i[7:]
        else:
            name = i
        if name in current_state_dict:
            if current_state_dict[name].shape == k.shape:
                new_ckpt[name] = k
        else:
            new_ckpt[name] = k
    return new_ckpt

def get_files(img_dir: str) -> Tuple[List[str], List[str], List[str]]:
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path: str) -> Tuple[List[str], List[str], List[str]]:
    """Finds the image files in a directory recursively

    Args:
        in_path: path in which the images files has to be searched
    """
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm' or ext == '.bmp':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    return img_files, mask_files, gt_files
