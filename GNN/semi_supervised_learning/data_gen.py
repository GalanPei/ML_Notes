import numpy as np
import random
import math
import matplotlib.pyplot as plt


class CaseGenerate(object):
    """
    Generate simple cases for testing the feasibility of our
    SSL algorithms.
    """
    def __init__(self, num_unlabeled: int, num_labeled: int) -> None:
        self.num_unlabled = num_unlabeled
        self.num_labeled = num_labeled
        self.total_num = self.num_labeled + self.num_unlabled

    def gen_simple_graph(self):
        x1_l = np.random.uniform(-1, 1, size=(self.num_labeled, 1))
        x2_l = np.random.uniform(-1, 1, size=(self.num_labeled, 1))
        x1_u = np.random.uniform(-1, 1, size=(self.num_unlabeled, 1))
        x2_u = np.random.uniform(-1, 1, size=(self.num_unlabeled, 1))
        f_l = np.zeros((self.num_labeled, 1))
        true_label = np.zeros((self.num_unlabeled, 1))
        for i in range(self.num_labeled):
            if x1_l[i, 0] + x2_l[i, 0] > 0:
                f_l[i, 0] = 1
        for i in range(self.num_unlabeled):
            if x1_u[i, 0] + x2_u[i, 0] > 0:
                true_label[i, 0] = 1
        labeled_data = np.hstack((x1_l, x2_l, f_l))
        unlabeled_data = np.hstack((x1_u, x2_u))
        return labeled_data, unlabeled_data, true_label

    def gen_synthetic_graph(self, GraphType='Multi'):
        theta_1 = np.random.uniform(
            np.pi, 7 / 4 * np.pi, size=(self.total_num, 1))
        theta_2 = np.random.uniform(-0.55 * np.pi,
                                    1 / 9 * np.pi, size=(self.total_num, 1))
        theta_3 = np.random.uniform(0, np.pi, size=(self.total_num, 1))
        r_1 = np.random.uniform(2.5, 3.5, size=(self.total_num, 1))
        r_2 = np.random.uniform(2.5, 3.5, size=(self.total_num, 1))
        r_3 = np.random.uniform(2.5, 3.5, size=(self.total_num, 1))
        x_1 = 4 + r_1 * np.cos(theta_1)
        y_1 = 4 + r_1 * np.sin(theta_1)
        x_2 = 3 + r_2 * np.cos(theta_2)
        y_2 = 6 + r_2 * np.sin(theta_2)
        x_3 = 6 + r_3 * np.cos(theta_3)
        y_3 = 5 + r_3 * np.sin(theta_3)
        labeledData = np.zeros((self.num_labeled, 3))
        sample = random.sample(list(range(self.total_num)), self.num_labeled)
        if GraphType == 'Multi':
            xData = np.vstack((x_1, x_2, x_3))
            yData = np.vstack((y_1, y_2, y_3))
            for i in range(self.num_labeled):
                if i % 3 == 1:
                    theta = theta_1[sample[i], 0]
                    labeledData[i, :] = np.array(
                        [4 + 3 * np.cos(theta), 4 + 3 * np.sin(theta), 0])
                if i % 3 == 2:
                    theta = theta_2[sample[i], 0]
                    labeledData[i, :] = np.array(
                        [3 + 3 * np.cos(theta), 6 + 3 * np.sin(theta), 1])
                if i % 3 == 0:
                    theta = theta_3[sample[i], 0]
                    labeledData[i, :] = np.array(
                        [6 + 3 * np.cos(theta), 5 + 3 * np.sin(theta), 2])
            unlabeledData = np.hstack((xData, yData))
            true_label = np.vstack((np.zeros((self.total_num, 1)), np.ones(
                (self.total_num, 1)), 2 * np.ones((self.total_num, 1))))
        if GraphType == 'Binary':
            xData = np.vstack((x_2, x_3))
            yData = np.vstack((y_2, y_3))
            for i in range(self.num_labeled):
                if i % 2 == 1:
                    theta = theta_2[sample[i], 0]
                    labeledData[i, :] = np.array(
                        [3 + 3 * np.cos(theta), 6 + 3 * np.sin(theta), 0])
                if i % 2 == 0:
                    theta = theta_3[sample[i], 0]
                    labeledData[i, :] = np.array(
                        [6 + 3 * np.cos(theta), 5 + 3 * np.sin(theta), 1])
            unlabeledData = np.hstack((xData, yData))
            true_label = np.vstack(
                (-1 * np.ones((self.total_num, 1)), np.ones((self.total_num, 1))))
        return labeledData, unlabeledData, true_label
