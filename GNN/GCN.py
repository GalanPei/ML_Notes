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


class GCNBase(nn.Module):
    """
    The implement of basic Graph Neural Network
    """
    def __init__(self, input_features: int, output_features: int, hidden_size: int, bias: list,
                 drop_out_rate: float = 0.5) -> None:
        """
        :param input_features:
        :param output_features
        :param hidden_size:
        :param drop_out_rate:
        """
        super(GCNBase, self).__init__()
        self.relu = F.relu
        self.dropout = nn.Dropout(drop_out_rate)
        weight0 = torch.empty(input_features, hidden_size)
        weight1 = torch.empty(hidden_size, output_features)
        self.weight0 = nn.Parameter(nn.init.xavier_normal_(weight0))
        self.weight1 = nn.Parameter(nn.init.xavier_normal_(weight1))
        self.bias = bias
        if bias[0]:
            bias0 = torch.empty(1, hidden_size)
            self.bias0 = nn.Parameter(nn.init.xavier_normal_(bias0))
        if bias[1]:
            bias1 = torch.empty(1, output_features)
            self.bias1 = nn.Parameter(nn.init.xavier_normal_(bias1))
        self.softmax = F.softmax

    def forward(self, X, A):
        """
        Forward propagation of GCN

        :param X: Input data
        :param A: Adjacent matrix
        :return:
        """
        assert X.shape[0] == self.weight0[0], \
            "The shape of input data should be same as the weight matrix"
        X = A @ X @ self.weight0
        if self.bias[0]:
            X += self.bias0
        X = self.relu(X)
        X = self.dropout(X)
        X = A @ X @ self.weight1
        if self.bias[1]:
            X += self.bias1
        X = self.relu(X)
        X = self.dropout(X)
        return X
