import torch
import torch.nn as nn
import torch.nn.init as init
from typing import List

def init_weights(modules: List[nn.Module]) -> None:
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

class SegmentNet1(nn.Module):
    def __init__(self, in_channel) -> None:
        super(SegmentNet1, self).__init__()
        self.last_conv: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.aspp1: nn.Sequential = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1))
        self.aspp2: nn.Sequential = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=12, padding=12), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1))
        self.aspp3: nn.Sequential = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=18, padding=18), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1))
        self.aspp4: nn.Sequential = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=24, padding=24), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1))

        init_weights(self.last_conv.modules())
        init_weights(self.aspp1.modules())
        init_weights(self.aspp2.modules())
        init_weights(self.aspp3.modules())
        init_weights(self.aspp4.modules())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        refine = self.last_conv(x)
        aspp1 = self.aspp1(refine)
        aspp2 = self.aspp2(refine)
        aspp3 = self.aspp3(refine)
        aspp4 = self.aspp4(refine)
        out = aspp1 + aspp2 + aspp3 + aspp4
        return out.permute(0, 2, 3, 1)