import torch
from torch import nn
import torch.nn.functional as F


class Aggregator(object):
    def __init__(self, inputs):
        self.inputs = inputs
        self.outputs = torch.zeros_like(inputs)

    def __call__(self):
        return self.outputs


class MeanAggregator(Aggregator):
    def __init__(self, inputs):
        super(MeanAggregator, self).__init__(inputs)

    def __call__(self):
        print("Mean aggregator is deployed!")
        self.outputs = torch.mean(input=self.inputs, dim=1)
        return self.outputs


class InductiveAggregator(Aggregator):
    def __init__(self, inputs):
        super(InductiveAggregator, self).__init__(inputs)

    def __call__(self):
        print("")
        return self.outputs


class LSTMAggregator(Aggregator):
    def __int__(self, inputs):
        super(LSTMAggregator, self).__init__(inputs)

    def __call__(self):
        print("")
        return self.outputs


class PoolingAggregator(Aggregator):
    def __int__(self, inputs):
        super(PoolingAggregator, self).__init__(inputs)

    def __call__(self):
        print("Pooling aggregator is deployed!")
        return self.outputs
