import numpy as np


class NaiveBayes:
    def __init__(self, data, input):
        self.data = data
        self.x = data[:, :-1]
        self.y = data[:, -1]
        self.input = input
        self.N = self.x.shape[0]
        self.n = self.x.shape[1]
        self.K = len(set(self.y))

    def train(self, l=1):
        label_set = set(self.y)
        prob = 0
        output = 0
        for label in iter(label_set):
            p_y = (np.sum(np.where(self.y == label, 1, 0)) + l) / (self.N + self.K * l)
            p_x = (np.sum(np.where((self.data[:, 0] == self.input[0]) & (self.data[:, -1] == label), 1, 0)) + l) / \
                  (np.sum(np.where(self.y == label, 1, 0)) + len(set(self.x[:, 0])) * l)
            for j in range(1, self.n):
                p_x *= (np.sum(np.where((self.data[:, j] == self.input[j]) & (self.data[:, -1] == label), 1, 0)) + l) / \
                       (np.sum(np.where(self.y == label, 1, 0)) + len(set(self.x[:, j])) * l)
            if p_y * p_x > prob:
                prob = p_y * p_x
                output = label
        return output


if __name__ == '__main__':
    data = np.array([[1, 'S', -1],
                     [1, 'M', -1],
                     [1, 'M', 1],
                     [1, 'S', 1],
                     [1, 'S', -1],
                     [2, 'S', -1],
                     [2, 'M', -1],
                     [2, 'M', 1],
                     [2, 'L', 1],
                     [2, 'L', 1],
                     [3, 'L', 1],
                     [3, 'M', 1],
                     [3, 'M', 1],
                     [3, 'L', 1],
                     [3, 'L', -1]])
    print(NaiveBayes(data, ['2', 'S']).train())
