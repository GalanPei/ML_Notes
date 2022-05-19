import torch
from torch.nn import functional as F
import numpy as np
from torch import nn
from d2l import torch as d2l


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


def gaussian(dim1, dim2):
    return torch.randn(dim1, dim2, requires_grad=True) * .1


def init_state(input_size, hidden_size, output_size):
    weight = []
    bias = []

    if isinstance(hidden_size, int):
        size_list = [input_size, hidden_size, output_size]
    elif isinstance(hidden_size, list):
        size_list = [input_size] + hidden_size + [output_size]

    for i in range(len(size_list) - 1):
        weight.append(gaussian(size_list[i], size_list[i + 1]))
        bias.append(torch.zeros(size_list[i + 1], requires_grad=True))

    return weight, bias


class MLPScratch(object):
    def __init__(self, input_size,
                 hidden_size,
                 output_size,
                 activation_func,
                 optimizer,
                 loss):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weight, self.bias = init_state(self.input_size, self.hidden_size, self.output_size)
        self.activation_func = activation_func
        self.optimizer = optimizer
        self.loss = loss

    def train(self, num_epochs):
        for i in range(num_epochs):
            pass

    def net(self, X):
        X = X.reshape((-1, self.input_size))
        for i in range(len(self.weight) - 1):
            W, b = self.weight[i], self.bias[i]
            X = self.activation_func(X @ W + b)
        return X @ self.weight[-1] + self.bias[-1]


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
