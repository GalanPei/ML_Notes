import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class DataTransform(object):
    """
    Represent the input data as graph.
    The detailed implement method is to transform the input data as
    different kinds of matrices including adjacent matrix, Graph
    Laplacian matrix and normalized Graph Laplacian.
    """

    def __init__(self, inputs: torch.Tensor):
        """
        :param inputs: input features
        """
        self.inputs = inputs
        # Dimension of input features
        self.n_dim = inputs.shape[0]
        self.adj_mat = torch.zeros(size=(self.n_dim, self.n_dim))
        self.diag_mat = torch.zeros(size=(self.n_dim, self.n_dim))
        self.graph_laplacian = torch.zeros(size=(self.n_dim, self.n_dim))
        self.normalized_laplacian = torch.zeros(size=(self.n_dim, self.n_dim))
        self.normalized_adj = torch.zeros(size=(self.n_dim, self.n_dim))

    def generate_adj_matrix(self, dist_fun) -> torch.Tensor:
        for i in range(self.n_dim):
            for j in range(i, self.n_dim):
                if i == j:
                    continue
                self.adj_mat[i][j] = dist_fun(self.inputs[i], self.inputs[j])
                self.adj_mat[j][i] = dist_fun(self.inputs[j], self.inputs[i])
        return self.adj_mat

    def generate_diag_matrix(self) -> torch.Tensor:
        for i in range(self.n_dim):
            self.diag_mat[i][i] = torch.sum(self.inputs[i])
        return self.diag_mat

    def generate_laplace_matrix(self) -> torch.Tensor:
        self.graph_laplacian = self.diag_mat - self.adj_mat
        return self.graph_laplacian

    def generate_normalized_laplacian(self) -> torch.Tensor:
        temp_diag = torch.tensor([self.diag_mat[i][i] ** (-0.5) \
                                  for i in range(self.n_dim)])
        self.normalized_laplacian = torch.eye(self.n_dim) + \
                                    temp_diag * self.adj_mat * temp_diag
        return self.normalized_laplacian

    def data_preprocessing(self, dist_fun) -> None:
        """
        Data pre-processing of input data. By this method, we will get
        adjacent matrix, diagonal matrix, graph laplacian matrix and
        normalized laplacian matrix of input data.

        :param dist_fun: Distance function used to generate adjacent matrix
        """
        self.generate_adj_matrix(dist_fun)
        self.generate_diag_matrix()
        self.generate_laplace_matrix()
        self.generate_normalized_laplacian()

    def neighborhood_sampling(self, node: int, sampling_type: str = "full",
                              threshold: float = 0., sampling_num: int = None):
        res = []
        for nd in self.adj_mat[node]:
            if nd != node and self.adj_mat[node][nd] > threshold:
                res.append(nd)
        if sampling_type == "full":
            return res
        elif sampling_type == "random":
            if len(res) <= sampling_num:
                return res
            choice = np.random.choice(res, size=sampling_num, replace=False)
            return list(choice)
