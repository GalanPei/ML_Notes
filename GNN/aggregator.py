import torch
from torch import nn
import torch.nn.functional as F


class Aggregator(object):
    def __init__(self, inputs):
        self.outputs = torch.zeros_like(inputs)

    def __call__(self):
        return self.outputs


class MeanAggregator(Aggregator):
    def __init__(self, inputs):
        super(MeanAggregator, self).__init__()


class LSTMAggregator(Aggregator):
    def __int__(self):
        super(LSTMAggregator, self).__init__()


class PoolingAggregator(Aggregator):
    def __int__(self):
        super(PoolingAggregator, self).__init__()
