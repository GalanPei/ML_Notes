import torch


def l2_loss(y_hat, y):
    return 1/2*float(torch.pow(y_hat - y, 2).sum())


class TorchPerceptron(object):
    def __init__(self):
        pass

    def train(self):
        pass
