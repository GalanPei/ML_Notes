import torch
from torch import nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, input_channels, mid_channels=None,
                 out_channels=None, use_1x1conv=True,
                 strides=1, act_fun=F.relu):
        super(Residual, self).__init__()
        if not mid_channels:
            mid_channels = input_channels
        if not out_channels:
            raise ValueError("Output channel is not given!")

        self.conv1 = nn.Conv2d(input_channels, mid_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(mid_channels, out_channels,
                               kernel_size=3, padding=1, stride=strides)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, out_channels,
                                   kernel_size=1, padding=1, stride=strides)
        else:
            self.conv3 = None

        self.batchnorm1 = nn.BatchNorm2d(mid_channels)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.act_fun = act_fun

    def forward(self, X):
        Y = self.act_fun(self.batchnorm1(self.conv1(X)))
        Y = self.batchnorm2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.act_fun(Y)
