"""
Softmax回归的实现。
ref: 动手学深度学习 v2
原作者: 李沐
link: https://courses.d2l.ai/zh-v2/
"""

import torch
from IPython import display
from d2l import torch as d2l
import softmax_regression.fashion_mnist_visual as fashion_mnist
import matplotlib.pyplot as plt
import pylab

batch_size = 256
train_iter, test_iter = fashion_mnist.load_data_fashion_mnist(batch_size)
num_inputs = 784  # 将28*28矩阵展平为784*1的向量
num_outputs = 10
lr = .1

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
num_epochs = 10


class Accumulator(object):
    """
    Accumulator 实例中创建了 2 个变量，用于分别存储正确预测的数量和预测的总数量.
    在`n`个变量上累加。
    """
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def softmax(X):
    """实现softmax"""
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


def net(X):
    """实现softmax回归模型"""
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


def cross_entropy(y_hat, y):
    """
    交叉熵函数
    :param y_hat: 预测值，表示形式为各个类别的预测概率
    :param y: 真实值，表示形式为 y_hat 中的概率索引
    :return: y_hat与y的交叉熵
    """
    return -torch.log(y_hat[[i for i in range(len(y_hat))], y])


def accuracy(y_hat, y):
    """
    计算预测正确的数量。
    :param y_hat:
    :param y:
    :return: 分类准确率
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        temp_y_hat = y_hat.argmax(axis=1)
    else:
        temp_y_hat = y_hat
    cmp = temp_y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度。"""
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期（定义见第3章）。"""
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(
                float(l) * len(y), accuracy(y_hat, y),
                y.size().numel())
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练准确率
    return metric[0] / metric[2], metric[1] / metric[2]


class Animator(object):
    """在动画中绘制数据。"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        # d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes,]
        self.config_axes = lambda: d2l.set_axes(self.axes[
            0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        print(x, y)
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, _b) in enumerate(zip(x, y)):
            if a is not None and _b is not None:
                plt.draw()
                plt.pause(0.001)
                self.X[i].append(a)


def train_ch3(net, train_iter, test_iter, loss, num_epochs: int, updater):
    """训练模型（定义见第3章）。"""
    assert num_epochs > 0, num_epochs
    # animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
    #                     legend=['train loss', 'train acc', 'test acc'])
    loss_list, acc_list, test_acc_list = [], [], []
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        train_loss, train_acc = train_metrics
        test_acc = evaluate_accuracy(net, test_iter)
        loss_list.append(train_loss)
        acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        # animator.add(epoch + 1, train_metrics + (test_acc,))

    x_list = [i for i in range(1, num_epochs + 1)]
    plt.plot(x_list, loss_list,
             x_list, acc_list,
             x_list, test_acc_list)
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert 0.7 < train_acc <= 1, train_acc
    assert 0.7 < test_acc <= 1, test_acc


def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)


def predict_ch3(net, test_iter, n=6):
    """预测标签（定义见第3章）。"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
# d2l.plt.show()
predict_ch3(net, test_iter)
pylab.show()
