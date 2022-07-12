import torch
from torch import nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, input_channels, mid_channels, out_channels, use_1x1conv=True, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, mid_channels, \
            kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, \
            kernel_size=3, padding=1, stride=strides)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, out_channels,\
                kernel_size=1, padding=1, stride=strides)

    def forward(self):
        pass
