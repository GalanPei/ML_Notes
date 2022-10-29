import torch
from torch import nn
import torch.nn.functional as F


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

        # Check out the input shape
        assert X.shape[0] == self.weight0[0], \
            "The shape of input data should be same as the weight matrix"
        # Propagation of first layer:
        #   X_1 = Relu(A * X * W_0)
        X = A @ X @ self.weight0
        if self.bias[0]:
            X += self.bias0
        X = self.relu(X)
        X = self.dropout(X)
        # Propagation of second layer:
        #   Z = softmax(A * X_1 * W_1)
        X = A @ X @ self.weight1
        if self.bias[1]:
            X += self.bias1
        Z = self.relu(X)
        Z = self.dropout(Z)
        return Z
