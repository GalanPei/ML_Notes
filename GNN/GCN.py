import torch
from torch import nn
import torch.nn.functional as F


class DataTransform(object):
    """
    Represent the input data as graph.
    The detailed implement method is to transform the input data as
    different kinds of matrices including adjacent matrix, Graph
    Laplacian matrix and normalized Graph Laplacian.
    """

    def __init__(self, inputs: torch.Tensor):
        self.inputs = inputs
        self.n_dim = inputs.shape[0]
        self.adj_mat = torch.zeros(size=(self.n_dim, self.n_dim))
        self.diag_mat = torch.zeros(size=(self.n_dim, self.n_dim))
        self.graph_laplacian = torch.zeros(size=(self.n_dim, self.n_dim))

    def generate_adj_matrix(self, dist_fun) -> torch.Tensor:
        for i in range(self.n_dim):
            for j in range(i, self.n_dim):
                self.adj_mat[i][j] = dist_fun(self.inputs[i], self.inputs[j])
                self.adj_mat[j][i] = dist_fun(self.inputs[j], self.inputs[i])
        return self.adj_mat

    def generate_diag_matrix(self):
        for i in range(self.n_dim):
            self.diag_mat[i][i] = torch.sum(self.inputs[i])

    def generate_laplace_matrix(self):
        temp_diag = torch.tensor([self.diag_mat[i][i] ** (-0.5) for i in range(self.n_dim)])
        self.graph_laplacian = torch.eye(self.n_dim) - temp_diag * self.adj_mat * temp_diag
        return self.graph_laplacian

    def normalized_laplacian(self):
        pass



class GCNBase(nn.Module):
    """
    The implement of basic Graph Neural Network
    """

    def __init__(self, input_features, drop_out_rate=0.5) -> None:
        super().__init__()
        self.relu = F.relu

    def forward(self):
        pass
