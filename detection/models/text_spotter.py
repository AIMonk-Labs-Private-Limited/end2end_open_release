import torch
import torch.nn as nn
from .model_utils import *
from typing import Any, Dict, Optional, Tuple

class TextSpotter(nn.Module):
    def __init__(self,config: Dict[str, Any]) -> None:
        """End to End Text spotting model class
        
        Args:
            config: Dictionary config for model creation
        """
        super(TextSpotter, self).__init__()
        self.config: Dict[str, Any] = config
        self.backbone_type: str = self.config['model_params']['backbone_params']['type']
        self.detector_type:str = self.config['model_params']['detector_params']['type']
        self.do_detection = get_check_flags(self.config['model_params'])
        self.initialize_models()

    def initialize_models(self) -> None:
        """Initialises different components of E2E model"""
        self.basenet: Optional[nn.Module] = get_backbone(self.backbone_type)(
                                                self.config['model_params']['backbone_params']) \
                                                if self.do_detection else None
        self.detector: Optional[nn.Module] = get_detector(self.detector_type)(
                                                self.config['model_params']) \
                                                if self.do_detection else None
        self.fix_detector = self.config['model_params'].get('fix_detector', False)
        if self.fix_detector:
            for param in self.basenet.parameters():
                param.requires_grad = False
            for param in self.detector.parameters():
                param.requires_grad = False
        self.update_model_status()

    def update_model_status(self):
        if self.fix_detector:
            self.basenet.eval()
            self.detector.eval()

    def forward(self, input_images: torch.Tensor, **kwargs: Any
               ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], 
                          Optional[torch.Tensor]]:
        """End to End forward pass
        
        Args:
            input_images: Input image tensor. Shape is (Batch_size, 3, Height, Width)
            **kwargs: Additional parameters for forward pass

        Returns:
            ret_heatmap: Tensor containing region, affinity, uppercase or symbol map. 
                          Shape is (Batch_size, (2 or 3 or 4), Out_height, Out_width)
            ret_word_direction: Tensor containing word direction map.
                                 Shape is (Batch_size, Out_height, Out_width, 2)
            ret_seg_map: Tensor containing word segmentation map.
                          Shape is (Batch_size, Out_height, Out_width, 1)
        """
        ret_heatmap, ret_word_direction, ret_seg_map, \
                            detector_features = None, None, None, None
        if self.do_detection:
            backbone_features = self.forward_basenet(input_images)
            detector_features = self.forward_detector(backbone_features)
            ret_heatmap, ret_word_direction, ret_seg_map = (detector_features[0], 
                                                            detector_features[1], 
                                                            detector_features[2])
        
        return (ret_heatmap, ret_word_direction, ret_seg_map)

    def forward_basenet(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass of image feature extractor
        
        Args:
            x: Input image tensor. Shape is (Batch_size, 3, Height, Width)

        Returns:
            out: Tuple containing feature map tensors at different strides
        """
        out = self.basenet(x)
        return out

    def forward_detector(self, x: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """Forward pass of FPN(kind of) network
        
        Args:
            x: Tuple containing feature map tensors at different scales
        
        Returns:
            out: Tuple containing different heatmaps like region, affinity, direction etc
        """
        out = self.detector(x)
        return out

    def fuse_modules(self) -> None:
        """Fuses the Conv, BN and Relu layer in backbone, detector and recognizer models"""
        if (self.basenet is not None) and hasattr(self.basenet, 'fuse_modules'):
            print("Fusing Backbone network layers...")
            self.basenet.fuse_modules()
        if (self.detector is not None) and hasattr(self.detector, 'fuse_modules'):
            print("Fusing Detector network layers...")
            self.detector.fuse_modules()
