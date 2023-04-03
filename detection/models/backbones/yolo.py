import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Union
from .utils import Focus, C3, SPP, YoloConv, initialize_weights_yolo

yolo_configs: Dict[str, Any] = {
    'small': {
        'focus': 32, 'conv1': 64, 'c3_1': (64, 1),
        'conv2': 128, 'c3_2': (128, 3),
        'conv3': 256, 'c3_3': (256, 3),
        'conv4': 384, 'spp': 480
    }
}

class yolo_small(torch.nn.Module):
    arch_config: Dict[str, Union[int, Tuple[int, int]]] = yolo_configs['small']
    outfea_dims: List[int] = [arch_config['spp'], arch_config['conv4'], arch_config['conv3'],
                              arch_config['conv2'], arch_config['conv1']]
    #should be same length as forward method output and should contain channel dim of each feature map.
    def __init__(self, config: Dict[str, Any]) -> None:
        super(yolo_small, self).__init__()
        pretrained: bool = config['pretrained']
        freeze: bool = config['freeze']
        if pretrained: raise ValueError("Pretrained model not available with yolo")
        self.focus: nn.Module = Focus(3, self.arch_config['focus'], k=3, s=1)
        self.conv1: nn.Module = YoloConv(self.arch_config['focus'], self.arch_config['conv1'], 3, 1)
        self.c3_1: nn.Module = C3(self.arch_config['conv1'], self.arch_config['c3_1'][0],
                                  self.arch_config['c3_1'][1]) #/2
        self.conv2: nn.Module = YoloConv(self.arch_config['c3_1'][0],
                                         self.arch_config['conv2'], 3, 2)
        self.c3_2: nn.Module = C3(self.arch_config['conv2'], self.arch_config['c3_2'][0],
                                  self.arch_config['c3_2'][1]) #/4
        self.conv3: nn.Module = YoloConv(self.arch_config['c3_2'][0],
                                         self.arch_config['conv3'], 3, 2) 
        self.c3_3: nn.Module = C3(self.arch_config['conv3'], self.arch_config['c3_3'][0],
                                  self.arch_config['c3_3'][1]) #/8
        self.conv4: nn.Module = YoloConv(self.arch_config['c3_3'][0],
                                         self.arch_config['conv4'], 3, 2)
        self.spp: nn.Module = SPP(self.arch_config['conv4'], self.arch_config['spp']) #this is dilated conv not maxpooling
        initialize_weights_yolo(self)

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        x = self.focus(x)
        x = self.c3_1(self.conv1(x))
        down_2 = x
        x = self.c3_2(self.conv2(x))
        down_4 = x
        x = self.c3_3(self.conv3(x))
        down_8 = x
        x = self.conv4(x)
        down_16 = x
        x = self.spp(x)
        down_16b = x
        return down_16b, down_16, down_8, down_4, down_2

    def fuse_modules(self) -> None:
        for _, mod_ins in self.named_modules():
            if isinstance(mod_ins, YoloConv):
                mod_ins.fuse_modules()
                mod_ins.forward = mod_ins.fuseforward
