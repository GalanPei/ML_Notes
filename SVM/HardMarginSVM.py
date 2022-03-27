import numpy as np


class HardSVM(object):
    def __init__(self, dataset):
        self.x = dataset[:, :-1]
        self.y = dataset[:, -1]

    def SMO(self):
        pass
