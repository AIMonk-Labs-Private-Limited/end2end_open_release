"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math
import torch
from typing import List, Tuple, Optional

def sort_rectangle2(poly: np.ndarray) -> Tuple[np.ndarray, int]:
    """Orders the given points in clockwise order

    Args:
        poly: Input box array of shape (4, 2)

    Retunrs:
        array: Clockwise ordered array of shape (4, 2)
        int: top left point's index in the result array
    """
    # First find the lowest point
    p_lowest = np.argmax(poly[:, 1])
    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        # if the bottom line is parallel to x-axis, then p0 must be the upper-left corner
        p0_index = np.argmin(np.sum(poly, axis=1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
    else:
        # find the point that sits right to the lowest point
        p_lowest_right = (p_lowest - 1) % 4
        p_lowest_left = (p_lowest + 1) % 4
        p_lowr_dist, p_lowl_dist = (np.linalg.norm(poly[p_lowest] - poly[p_lowest_right]), 
                                    np.linalg.norm(poly[p_lowest] - poly[p_lowest_left]))
        if p_lowr_dist > p_lowl_dist:
            p3_index = p_lowest
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], 0
        else:
            p2_index = p_lowest
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index + 1) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], 0

def sort_box_with_direction(direction_map: np.ndarray, crop_region_flag: np.ndarray,
                            box: np.ndarray) -> np.ndarray:
    """Orders the four corners of box in clockwise direction with information from direction map

    Args:
        direction_map: Crop containing the direction map values. Shape is (X, Y, 2)
        crop_region_flag: Crop containing bool values indicating region to consider direction values
                           Shape is (X, Y)
        box: Array containing box coordinates. Shape is (4, 2)

    Returns:
        Array containing box coordinates ordered according to direction map. Shape is (4, 2)
    """
    y_ordered = box[np.argsort(box[:, 1]), :]
    top2_x_ordered = y_ordered[:2, :][np.argsort(y_ordered[:2, 0]), :]
    bottom2_x_ordered_desc = y_ordered[2:, :][np.argsort(y_ordered[2:, 0])[::-1], :]
    sorted_box = np.concatenate([top2_x_ordered, bottom2_x_ordered_desc], 0)
    vec_angles = direction_map[crop_region_flag, :]
    dir_vector = np.array([sorted_box[1] - sorted_box[0], 
                           sorted_box[2] - sorted_box[1], 
                           sorted_box[3] - sorted_box[2], 
                           sorted_box[0] - sorted_box[3]])
    vec_angles = vec_angles / np.linalg.norm(vec_angles, axis=-1, keepdims=True) #normalising 
    dir_vector = dir_vector / np.linalg.norm(dir_vector, axis=-1, keepdims=True) #normalising
    distance = (dir_vector[np.newaxis, :, :] * vec_angles[:, np.newaxis, :]).sum(-1) #dot product
    distance = 1 - distance #distance in the range of (0, 2)
    new_dist = distance.mean(0)
    min_index = np.argmin(new_dist)
    tl, tr, br, bl = ((min_index + 1) % 4, (min_index + 2) % 4 , 
                      (min_index + 3) % 4, min_index)
    box = sorted_box[[tl, tr, br, bl], :]
    return box

def decode_maps_cpu(textmap: Optional[np.ndarray], linkmap: Optional[np.ndarray], 
                symbol_map: Optional[np.ndarray], 
                direction: Optional[np.ndarray], decode_type: str,
                link_threshold: float, region_threshold: float,
                symbol_threshold: float, dilate_with_symbol: bool) -> List[torch.Tensor]:
    """Decodes given array to give final text box locations in CPU

    Args:
        textmap: Region map array. Shape is (X, Y)
        linkmap: Affinity map array. Shape is (X, Y)
        symbol_map: Symbol segmentation map array. Shape is (X, Y)
        direction: Direction map array. Shape is (X, Y, 2)
        decode_type: Whether to decode using (textmap, linkmap) combination or segmap
        link_threshold: Threshold for binarising linkmap
        region_threshold: Threshold for binarising textmap
        symbol_threshold: Threshold for binarising symbol_map
        dilate_with_symbol: Whether to add symbol map regions in final array to be dilated

    Returns:
        List containing a tensor of shape (4,2) for each predicted text box
    """
    assert decode_type in ['heatmap']
    if decode_type == "heatmap":
        img_h, img_w = textmap.shape
        _, text_score = cv2.threshold(textmap, region_threshold, 1, 0)
        _, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)
        if symbol_map is not None:
            _, symbol_score = cv2.threshold(symbol_map, symbol_threshold, 1, 0)
            text_score_comb = np.clip(text_score + link_score + symbol_score, 0, 1)
            if dilate_with_symbol:
                text_dil_flag = np.logical_or(symbol_score == 1,
                                       np.logical_and(link_score==0, text_score==1))
            else:
                text_dil_flag = np.logical_and(link_score==0, text_score==1)
        else:
            text_score_comb = np.clip(text_score + link_score, 0, 1)
            text_dil_flag = np.logical_and(link_score==0, text_score==1)

    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                                        text_score_comb.astype(np.uint8), 
                                        connectivity=4)
    sort_box = []
    for k in range(1,nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2) #kernel size used for dilation and is determined exmpirically
        #Extending axis alinged box according to determined kernel size
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        #boundary check
        if sx < 0 : sx = 0
        if sy < 0 : sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        label_flag = labels[sy:ey, sx:ex] == k
        # 0.7 corresponds to that threshold for the textmap
        if (decode_type == "heatmap" and 
                    np.max(textmap[sy:ey, sx:ex][label_flag]) < 0.7):
            continue
        #Setting the pixels in the connected blob that belongs to label k and is only a character region
        dilation_inp = (label_flag* text_dil_flag[sy:ey, sx:ex]).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        crop_seg = cv2.dilate(dilation_inp, kernel)
        crop_non_zero_index = np.array(np.where(crop_seg!=0))
        #Changing the points from crop coordinate system to image coordinate system
        non_zero_index = crop_non_zero_index + np.array([sy, sx]).reshape(2, 1)
        np_contours = np.roll(non_zero_index,1,axis=0).transpose().reshape(-1,2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)
        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            #TODO: check this logic and remove if not needed
            l, r = min(np_contours[:,0]), max(np_contours[:,0])
            t, b = min(np_contours[:,1]), max(np_contours[:,1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4-startidx, 0)
        #TODO: check the above step effect and remove if not needed
        box = np.array(box)
        if direction is not None:
            if decode_type == "heatmap":
                #consider direction values only from regions with high confidence
                direction_crop_flag = np.logical_and(label_flag, 
                                                    textmap[sy:ey, sx:ex] > 0.4)
            else:
                direction_crop_flag = np.logical_and(label_flag,
                                                    text_dil_flag[sy:ey, sx:ex] > 0.95)
            direction_crop = direction[sy:ey, sx:ex]
            box = sort_box_with_direction(direction_crop, direction_crop_flag,
                                          box)
        else:
            box = sort_rectangle2(box)[0]
        sort_box.append(torch.from_numpy(box))
    return sort_box

def get_boxes_cpu(heatmap: Optional[torch.Tensor], word_dir_map: Optional[torch.Tensor],
            link_threshold: float, region_threshold: float, 
            symbol_threshold: float, dilate_with_symbol: bool,
            ) -> Tuple[List[torch.Tensor], Tuple[int, int], str]:
    """Uses the given tensor to run text box location decoding algorithm in CPU

    Args:
        heatmap: Tensor containing region map, affinity map, symbol map. Shape is (1, 2 or 3, X, Y)
        word_dir_map: Tensor containing direction map. Shape is (1, X, Y, 2)
        link_threshold: Threshold for binarising affinity map
        region_threshold: Threshold for binarising region map
        symbol_threshold: Threshold for binarising symbol_map
        dilate_with_symbol: Whether to add symbol map regions in final array to be dilated

    Returns:
        Tuple containing List of box tensors, decode map shape and decoding algorithm type
    """
    direction = None
    score_symbol = None
    if word_dir_map is not None:
        direction = word_dir_map[0, ...].cpu().data.numpy()
    if heatmap is not None and (heatmap.shape[1] in [2,3]):
        #using heatmaps and (word_dir_map if present) to get rotated boxes
        decode_type = "heatmap"
        score_text = heatmap[0,0,:,:].cpu().data.numpy()
        score_link = heatmap[0,1,:,:].cpu().data.numpy()
        if heatmap.shape[1] == 3:
            score_symbol = heatmap[0,2,:,:].cpu().data.numpy()
        output_shape = (score_text.shape[0], score_text.shape[1])
    else:
        raise ValueError("region+affinity map should"
                        " be predicted for decoding boxes. Use Valid Trained Model!")
    detected_boxes = decode_maps_cpu(score_text, score_link, score_symbol, direction,
                                decode_type, link_threshold,
                                region_threshold, symbol_threshold, dilate_with_symbol)
    return detected_boxes, output_shape, decode_type

def adjustResultCoordinates(polys: List[torch.Tensor], ratio_w: float, ratio_h: float, 
                            ratio_net: float = 2.0):
    """Scales the box location from output heatmap scale to input image scale

    Args:
        polys: List of torch tensors containing the boxes in heatmap scale
        ratio_w: ratio between input image width and original image width
        ratio_h: ratio between input image height and original image height
        ratio_net: Downsampling factor between network input and output

    Returns:
        polys: List of tensors containing the boxes in original image scale
    """
    if len(polys) > 0:
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= torch.tensor([ratio_w * ratio_net, ratio_h * ratio_net])
    return polys
