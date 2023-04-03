from .backbones import *
from .detectors import *
from typing import Dict, Any, Union, Tuple

mapping: Dict[str, Dict[str, Any]] = {
    'backbone_mapping': {
        'yolo_small': yolo_small
    },
    'detector_mapping': {
        'craft3': CRAFT3
    }
}

def get_backbone(typ: str) -> Union[yolo_small]:
    return mapping['backbone_mapping'][typ]

def get_detector(typ: str) -> Union[CRAFT3]:
    return mapping['detector_mapping'][typ]

def get_check_flags(config: Dict[str, Any]) -> Tuple[bool]:
    """Check different config parameters and add additional ones for model construction
    
    Args:
        config: Dictionary config for training and model construction

    Returns:
        do_detection: Whether to run detection network
    """
    do_detection = True
    config['detector_params']['in_map_dims'] = \
        get_backbone(config['backbone_params']['type']).outfea_dims
    return do_detection
