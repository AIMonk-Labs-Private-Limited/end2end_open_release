from .segment_net import SegmentNet1
import torch
import torch.nn as nn
from typing import Union, Tuple, Dict, List
segmentation_mapping = {
    'segment_1': SegmentNet1
}

def get_segmentation_net(typ: str, in_dims: int) -> Union[SegmentNet1]:
    return segmentation_mapping[typ](in_dims)

def get_activation(typ: str) -> nn.Module:
    if typ.lower() == "relu":
        return nn.ReLU(inplace=True)
    elif typ.lower() == "silu":
        return nn.SiLU(inplace=True)
    else:
        raise ValueError("Activation type {} not understood".format(typ))

def get_craft_map_module(in_channel_dim: int, heatmap_conv_dims: List[int],
                         act_typ: str = "relu") -> nn.Sequential:
    heatmap_layers: List[nn.Module] = []
    in_dims = in_channel_dim
    for num, out_dims in enumerate(heatmap_conv_dims):
        kernel_size = 1 if num > (len(heatmap_conv_dims) - 3) else 3 #last two convs are 1x1
        heatmap_layers.append(nn.Conv2d(in_dims, out_dims,
                              kernel_size=kernel_size, padding = 1 if kernel_size == 3 else 0))
        if (num <= len(heatmap_conv_dims)-2):
            heatmap_layers.append(get_activation(act_typ))
        in_dims = out_dims
    conv_cls = nn.Sequential(*heatmap_layers)
    return conv_cls

