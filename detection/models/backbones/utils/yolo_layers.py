import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from .common import fuse_conv_and_bn

def initialize_weights_yolo(model: nn.Module) -> None:
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True

def autopad(k: int, p: Optional[int] = None) -> int:  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class YoloConv(nn.Module):
    # Standard convolution
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: Optional[int] = None,
                 g: int = 1, act: bool = True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(YoloConv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else \
                        (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))

    def fuse_modules(self):
        self.conv = fuse_conv_and_bn(self.conv, self.bn)
        delattr(self, 'bn')

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1: int, c2: int, shortcut: bool = True,
                 g: int = 1, e: float = 0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = YoloConv(c1, c_, 1, 1)
        self.cv2 = YoloConv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True,
                 g: int = 1, e: float = 0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = YoloConv(c1, c_, 1, 1)
        self.cv2 = YoloConv(c1, c_, 1, 1)
        self.cv3 = YoloConv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class SPP(nn.Module):
    # Spatial pyramid pooling through dilation convolution
    def __init__(self, c1: int, c2: int, k: Tuple[int, int, int] = (3, 7, 11)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = YoloConv(c1, c_, 1, 1)
        self.cv2 = YoloConv(c_ * (len(k) + 1), c2, 1, 1)
        self.m1 = nn.Conv2d(c_, c_, 3, padding=k[0], dilation=k[0])
        self.m2 = nn.Conv2d(c_, c_, 3, padding=k[1], dilation=k[1])
        self.m3 = nn.Conv2d(c_, c_, 3, padding=k[2], dilation=k[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        x2 = torch.cat([x, self.m1(x), self.m2(x), self.m3(x)], 1)
        x3 = self.cv2(x2)
        return x3

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1,p: Optional[int] = None,
                 g: int = 1, act: bool = True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = YoloConv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))



