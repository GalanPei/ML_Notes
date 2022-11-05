import torch
from torch import nn
import matplotlib.pyplot as plt


def rbf_kernel(x1: torch.Tensor, x2: torch.Tensor, l=1., sig=1.):
    return l * torch.exp(torch.cdist(x1, x2) / 2 / sig)


class GaussianProcess(object):
    def __init__(self, x_test: torch.Tensor,
                 y_test: torch.Tensor,
                 x_train: torch.Tensor,
                 kernel=rbf_kernel) -> None:
        self.x_test = x_test
        self.y_test = y_test
        if x_test.ndim == 1:
            self.x_test = x_test.reshape(-1, 1)
        if y_test.ndim == 1:
            self.y_test = y_test.reshape(-1, 1)
        self.kernel = kernel
        self.x_train = x_train
        if x_train.ndim == 1:
            self.x_train = x_train.reshape(-1, 1)
        self.y_train = None
        self.K_xx = self.kernel(self.x_test, self.x_test)
        self.K_xs = self.kernel(self.x_test, self.x_train)
        self.K_sx = self.kernel(self.x_train, self.x_test)
        self.k_ss = self.kernel(self.x_train, self.x_train)

    def mean(self):
        self.y_train = self.K_sx @ torch.linalg.inv(self.K_xx) @ self.y_test
        return self.y_train

    def var(self):
        return self.K_ss - self.K_sx * torch.linalg.inv(self.K_xx) * self.K_xs

    def regression(self) -> None:
        self.mean()

    def append(self, x_train, y_train):
        pass

    def plot(self):
        plt.scatter(self.x_test, self.y_test, label="given data")
        plt.scatter(self.x_train, self.y_train, label="predict")
        plt.legend()
        plt.show()
