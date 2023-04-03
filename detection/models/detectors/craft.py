import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
from ..backbones import C3, initialize_weights_yolo, YoloConv
from .detector_utils import get_segmentation_net, get_craft_map_module

class CRAFT3(nn.Module):
    """
    Yolo layers based detection network
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        super(CRAFT3, self).__init__()
        self.config = config
        self.inp_map_dims: List[int] = config['detector_params']['in_map_dims']
        self.fpn_dims: List[int] = config['detector_params']['fpn_out_dim']
        self.upconv1: nn.Module = C3(self.inp_map_dims[0]+self.inp_map_dims[1], self.fpn_dims[0],
                                     shortcut=False)
        self.upconv2: nn.Module = C3(self.inp_map_dims[2]+self.fpn_dims[0], self.fpn_dims[1],
                                     shortcut=False)
        self.upconv3: nn.Module = C3(self.inp_map_dims[3]+self.fpn_dims[1], self.fpn_dims[2],
                                     shortcut=False)
        self.upconv4: nn.Module = C3(self.inp_map_dims[4]+self.fpn_dims[2], self.fpn_dims[3],
                                     shortcut=False)

        self.heatmap_conv_dims: List[int] = (config['detector_params']['heatmap_conv_dim']
                                                + [config['detector_params']['heatmap_out_dim']])
        self.conv_cls: Optional[nn.Sequential] = None
        self.seg_cls: Optional[nn.Module] = None
        if config['detector_params']['heatmap_out_dim'] > 0:
            self.conv_cls = get_craft_map_module(self.fpn_dims[-1], self.heatmap_conv_dims,
                                                 act_typ = "silu")
        #only if segmentation map is needed seg_cls is created
        if config['detector_params']['heatmap_channel_list'][2] > 0:
            self.seg_cls = get_segmentation_net(
                                config['detector_params']['segmentation_params']['type'], 
                                self.fpn_dims[-1])
        self.ocr_path = False
        self.ocr_fpn_dims: Optional[List[int]] = None
        if 'sep_ocr_path' in config['detector_params']: #for backward compat
            sep_ocr_path = config['detector_params']['sep_ocr_path'] 
        else:
            sep_ocr_path = config['sep_ocr_path']
        if (config['do_ocr'] and sep_ocr_path): 
            self.ocr_path = True
            self.ocr_fpn_dims = config['detector_params']['ocr_fpn_out_dim']
            self.upconv1_ocr: nn.Module = C3(self.inp_map_dims[0]+self.inp_map_dims[1], 
                                             self.ocr_fpn_dims[0], shortcut=False)
            self.upconv2_ocr: nn.Module = C3(self.inp_map_dims[2]+self.ocr_fpn_dims[0],
                                             self.ocr_fpn_dims[1], shortcut=False)
            self.upconv3_ocr: nn.Module = C3(self.inp_map_dims[3]+self.ocr_fpn_dims[1], 
                                             self.ocr_fpn_dims[2], shortcut=False)        
            self.upconv4_ocr: nn.Module = C3(self.inp_map_dims[4]+self.ocr_fpn_dims[2], 
                                             self.ocr_fpn_dims[3], shortcut=False)

        initialize_weights_yolo(self)
        self.out_fea_dims: List[int] = (config['detector_params']['heatmap_channel_list'] 
                                        + [self.fpn_dims[3], self.ocr_fpn_dims[-1] \
                                                             if self.ocr_fpn_dims is not None \
                                                             else self.fpn_dims[-1],
                                                             self.inp_map_dims[4]]) #dimensions of forward output of this module

    def forward(self, feature_maps: Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],torch.Tensor,torch.Tensor,torch.Tensor]:
        reg_aff_map, word_dir_map, seg_map = None, None, None
        assert len(feature_maps) == 5, "Currently this class needs 5 feature maps from backbone for processing"

        y = self.upconv1(torch.cat([feature_maps[0], feature_maps[1]], dim=1)) #[0], [1] is H/16, W/16
        y_ocr = self.upconv1_ocr(torch.cat([feature_maps[0], feature_maps[1]], 
                                            dim=1)) if self.ocr_path else y

        y = F.interpolate(y, size=feature_maps[2].size()[2:], mode='bilinear', align_corners=False)
        y_ocr = F.interpolate(y_ocr, size=feature_maps[2].size()[2:], mode='bilinear', 
                              align_corners=False) if self.ocr_path else y

        y = self.upconv2(torch.cat([y, feature_maps[2]], dim=1)) #[2] is H/8, W/8
        y_ocr = self.upconv2_ocr(torch.cat([y_ocr, feature_maps[2]],
                                            dim=1)) if self.ocr_path else y

        y = F.interpolate(y, size=feature_maps[3].size()[2:], mode='bilinear', align_corners=False)
        y_ocr = F.interpolate(y_ocr, size=feature_maps[3].size()[2:], mode='bilinear',
                              align_corners=False) if self.ocr_path else y

        y = self.upconv3(torch.cat([y, feature_maps[3]], dim=1))  #[3] is H/4, W/4
        y_ocr = self.upconv3_ocr(torch.cat([y_ocr, feature_maps[3]], dim=1)) if self.ocr_path else y

        y = F.interpolate(y, size=feature_maps[4].size()[2:], mode='bilinear', align_corners=False)
        y_ocr = F.interpolate(y_ocr, size=feature_maps[4].size()[2:], mode='bilinear',
                              align_corners=False) if self.ocr_path else y

        feature = self.upconv4(torch.cat([y, feature_maps[4]], dim=1))  #[4] is H/2, W/2
        feature_ocr = self.upconv4_ocr(torch.cat([y_ocr, feature_maps[4]], dim=1)) if self.ocr_path else feature
        if self.conv_cls is not None:
            y = self.conv_cls(feature)
            if self.out_fea_dims[0] > 0: #extract (region+affinity) maps
                s_index, e_index = 0, self.out_fea_dims[0] 
                reg_aff_map = y[:, s_index:e_index, :, :].contiguous()
            if self.out_fea_dims[1] > 0: #extract word direction maps
                s_index, e_index = self.out_fea_dims[0], self.out_fea_dims[0]+self.out_fea_dims[1]
                word_dir_map = y[:, s_index:e_index, :, :].permute(0, 2, 3, 1).contiguous()
        if self.seg_cls is not None: seg_map = self.seg_cls(feature)
        return reg_aff_map, word_dir_map, seg_map, feature, feature_ocr, feature_maps[4]

    def fuse_modules(self) -> None:
        "Fuses the Conv, BN and Relu layer for inference"
        for _, mod_ins in self.named_modules():
            if isinstance(mod_ins, YoloConv):
                mod_ins.fuse_modules()
                mod_ins.forward = mod_ins.fuseforward
