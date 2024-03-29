import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def epsilon_weight(t, epsilon, dim):
    if t < epsilon:
        return 1 / (epsilon ** dim)
    else:
        return 0


def gaussian_weight(t, epsilon, dim):
    return 1 / math.sqrt(2 * math.pi) / (epsilon ** dim) * math.exp(
        -t ** 2 / 2 / epsilon ** 2)


class LPA(object):
    def __init__(self, labeled_data, unlabeled_data, type='Multi'):
        """_summary_
        Initialize the class
        Args:
            `labeled_data` (`np.ndarray`): labeled data for learning, shape[l, d]: l: number of labeled    
                                           data, d: dimension of data
            `unlabeled_data` (`np.ndarray`): unlabeled data for learning, shape[u, d]: u: number of 
                                             unlabeled data, d: dimension of data
            `type` (`str`, optional): Defaults to 'Multi'.
        """
        self.labeled_data = labeled_data
        self.unlabeled_data = unlabeled_data
        self.num_sample = unlabeled_data.shape[0]
        self.num_labeled = labeled_data.shape[0]
        self.type = type
        self.dimension = unlabeled_data.shape[1]

    def weight_matrix(self, epsilon, weight_fun):
        """
        Build the weight matrix according to the type of weight function
        Args:
            epsilon: _description_
            weight_fun: weight function

        Returns:
            weight matrix of the graph
        """
        learning_data = np.vstack(
            (self.labeled_data[:, 0:-1], self.unlabeled_data))
        # Initialize the weight matrix, shape[l+u, l+u]
        mat_W = np.zeros((self.num_labeled + self.num_sample,
                          self.num_labeled + self.num_sample), np.float32)
        for i in range(self.num_labeled + self.num_sample):
            for j in range(self.num_labeled + self.num_sample):
                mat_W[i, j] = weight_fun(np.linalg.norm(
                    learning_data[i, :] - learning_data[j, :]),
                    epsilon)
        return mat_W

    def propagation_matrix(self, epsilon, weight_fun):
        """
        Build the normalized propagation matrix
        :param epsilon:
        :param weight_fun: weight function used for calculating distance
        :return: propagation matrix  P = D^{-1/2}WD^{-1/2}
        """
        mat_W = self.weight_matrix(epsilon, weight_fun)
        # Initialize the square root of inverse of the degree matrix, shape[l+u, l+u]
        inv_sqrt_D = np.zeros((self.num_labeled + self.num_sample,
                               self.num_labeled + self.num_sample), np.float32)
        # Build D^{-1/2}
        for i in range(self.num_labeled + self.num_sample):
            inv_sqrt_D[i, i] = 1 / math.sqrt(np.sum(mat_W[i, :]))
        # P = D^{-1/2}WD^{-1/2}
        matPropagation = np.linalg.multi_dot([inv_sqrt_D, mat_W, inv_sqrt_D])
        return matPropagation

    def label_propagation_improved(self,
                                   epsilon,
                                   weight_fun=gaussian_weight,
                                   alpha: float = 0.5,
                                   max_iteration: int = 1e4,
                                   tol: float = 1e-5):
        """
        Label Propagation Algorithm for multi-classification
        refer to: Dengyong Zhou, Olivier Bousquet, Thomas N Lal, Jason Weston, and Bernhard Schölkopf.
        Learning with local and global consistency. In Advances in neural information processing systems,
        pages 321-328, 2004.\n
        Args:
            `epsilon`:
            `weight_fun` (optional): Defaults to gaussian_weight.
            `alpha` (`float`, optional): learning parameter. Defaults to 0.5.
            `max_iteration` (`int`, optional): Defaults to 1e4.
            `tol` (`float`, optional): Defaults to 1e-5.

        Returns:
            vector_label: predicted label vector
            full_label:
        """
        labels = self.labeled_data[:, -1]  # Set of labels of labeled data
        label_list = np.unique(labels)  # Set of labels
        l = self.labeled_data.shape[0]
        u = self.unlabeled_data.shape[0]
        y = label_list.shape[0]
        # Initialize the predicted label vector
        F0 = np.zeros((l + u, y), np.float32)
        for i in range(l):
            F0[i, int(labels[i])] = 1
        full_label = F0
        # Create the propagation matrix
        S = self.propagation_matrix(epsilon, weight_fun)
        for iter in range(int(max_iteration)):
            F_old = full_label
            full_label = alpha * np.dot(S, full_label) + (1 - alpha) * F0
            if np.linalg.norm(full_label - F_old) < tol:
                break
        vec_label = np.argmax(full_label, axis=1)
        hat_F = np.zeros((vec_label.shape[0], 1))
        if self.type == 'Binary':
            for i in range(vec_label.shape[0]):
                hat_F[i, 0] = -1 * full_label[i, 0] + full_label[i, 1]
            return vec_label, hat_F
        else:
            return vec_label, full_label

    def labelPropOri(self,
                     epsilon,
                     weight_fun=gaussian_weight,
                     max_iteration: int = 1e4,
                     tol: float = 1e-5):
        """
        Label Propagation Algorithm for binary classification
        refer to: Xiaojin Zhu. Semi-supervised learning with graphs. Carnegie Mellon University, 2005.\n
        Args:
            `epsilon`: scale parameter
            `weight_fun` (`str`, optional): weight function. Defaults to 'Gaussian'.
            `max_iteration` (`int, optional): max iteration steps. Defaults to 1e4.
            `tol` (`float`, optional): tolerance. Defaults to 1e-5.

        Returns:
            vec_label: predicted label vector $\ell_i \in \{-1, 1\}$
        """
        l = self.labeled_data.shape[0]
        u = self.unlabeled_data.shape[0]
        f_l = self.labeled_data[:, -1]
        f_l = f_l.reshape((self.num_labeled, 1))
        for i in range(l):
            if f_l[i, 0] < 1e-5:
                f_l[i, 0] = -1
            else:
                f_l[i, 0] = 1
        learning_data = np.vstack(
            (self.labeled_data[:, 0:-1], self.unlabeled_data))
        mat_W = np.zeros((l + u, l + u))
        inv_D = np.zeros((l + u, l + u))
        for i in range(l + u):
            for j in range(l + u):
                mat_W[i, j] = weight_fun(np.linalg.norm(learning_data[i, :] - learning_data[j, :]),
                                         epsilon)
            inv_D[i, i] = 1 / np.sum(mat_W[i, :])
        mat_Puu = np.dot(inv_D[l:l + u, l:l + u], mat_W[l:l + u, l:l + u])
        mat_Pul = np.dot(inv_D[l:l + u, l:l + u], mat_W[l:l + u, 0:l])
        for i in range(int(max_iteration)):
            f_old = f_u
            f_u = np.dot(mat_Puu, f_u) + np.dot(mat_Pul, f_l)
            # Recurrence expression of the iteration
            if np.linalg.norm(f_u - f_old) < tol:
                break
        full_label = np.vstack((f_l, f_u))
        vec_label = np.sign(full_label)
        return vec_label, full_label

    def accuracy(self, vec_label, true_label):
        """
        Calculate the accuracy of LPA\n
        Args:
            true_label:
        Returns:
            accuracy $\in [0, 1]$
        """
        error_num = 0
        if self.type == 'Binary':
            vec_label = 2 * vec_label - 1
        for i in range(self.num_sample):
            if np.abs(vec_label[i + self.num_labeled] - true_label[i, 0]) > 1e-5:
                error_num += 1
        return 1 - error_num / self.num_sample

    def show_result(self, vec_label):
        """
        Plot the scatter figure of result\n
        Args:
            vec_label (`np.ndarray`): predicted label vector, shape[l+u, :]
        """
        vec_label = vec_label.reshape((self.num_labeled + self.num_sample, 1))
        f_l = vec_label[0:self.num_labeled, :]
        f_u = vec_label[self.num_labeled:(
            self.num_sample + self.num_labeled), :]
        learning_data = np.vstack(
            (self.labeled_data[:, 0:-1], self.unlabeled_data))
        colors = list(mcolors.TABLEAU_COLORS.keys())
        # Plot the unlabeled nodes as hollow circles
        for i in range(self.num_sample):
            plt.plot(self.unlabeled_data[i, 0],
                     self.unlabeled_data[i, 1],
                     linewidth=0.1,
                     marker='o',
                     markersize=2,
                     color=mcolors.TABLEAU_COLORS[colors[int(f_u[i])]], markerfacecolor='white')
        # Plot the labeled nodes as filled triangles
        for i in range(self.num_labeled):
            plt.plot(self.labeled_data[i, 0],
                     self.labeled_data[i, 1],
                     '^',
                     markersize=8,
                     color='black',
                     markerfacecolor=mcolors.TABLEAU_COLORS[colors[int(f_l[i])]])
        # plt.xlabel(r'$x_1$', fontsize=14)
        # plt.ylabel(r'$x_2$', fontsize=14)
        # plt.show()

    def plot_weight(self, vec_label):
        fig = plt.figure()
        vec_label = vec_label.reshape((self.num_sample + self.num_labeled, 1))
        # ax3d = Axes3D(fig)
        ax3d = fig.add_subplot(111, projection='3d')
        f_u = vec_label[self.num_labeled:(self.num_labeled + self.num_sample)]
        x = self.unlabeled_data[:, 0]
        y = self.unlabeled_data[:, 1]
        ax3d.plot_trisurf(x, y, f_u[:, 0], cmap=cm.coolwarm)
        # plt.show()

    def connectivity(self, epsilon, weight_fun='Epsilon'):
        """The connectivity of the graph"""
        mat_W = self.weight_matrix(epsilon, weight_fun)
        sum_weight = np.sum(mat_W, axis=1) - mat_W[0, 0]
        num_connect = np.sum(np.where(sum_weight > 1e-10, 1, 0))
        return num_connect / (self.num_labeled + self.num_sample)
