import numpy as np
import matplotlib.pyplot as plt


class Perceptron(object):
    def __init__(self, dataset):
        self.input = dataset[:, :-1]
        self.label = dataset[:, -1]
        self.dim = self.input.shape[1]
        self.num = self.input.shape[0]

    def train(self, eta=0.1, max_iter=1000):
        """
        Train a simple Perceptron algorithm

        :param eta: Learning rate of gradient descent
        :param max_iter: Max iteration steps
        :return: w, b the result of Hyper-parameters
        """
        w = np.zeros((1, self.dim))
        b = 0
        for _ in range(max_iter):
            wrong_count = 0
            for i in range(self.num):
                x = self.input[i, :]
                y = self.label[i]
                y_hat = np.sign(w @ x + b)
                if y_hat != y:
                    wrong_count += 1
                    w += eta*y*x
                    b += eta*y
            if wrong_count == 0:
                break
        return w, b

    def plot_result(self, eta=0.1, max_iter=1000):
        if self.dim >= 4:
            # If the dim of the data is more than 4, cannot plot
            raise Exception('Dimension should be less than 4!')
        w, b = self.train(eta, max_iter)
        if self.dim == 2:
            plt.scatter(self.input[:, 0], self.input[:, 1])
            x = np.arange(np.min(self.input[:, 0]), np.max(self.input[:, 0]), 0.1)
            y = np.arange(np.min(self.input[:, 1]), np.max(self.input[:, 1]), 0.1)
            x, y = np.meshgrid(x, y)
            z = w[0, 0] * x + w[0, 1] * y + b
            plt.contour(x, y, z, 0)
            plt.show()


if __name__ == '__main__':
    dataset = np.array([[3, 3, 1],
                        [4, 3, 1],
                        [1, 1, -1]])
    Per = Perceptron(dataset)
    Per.plot_result()
