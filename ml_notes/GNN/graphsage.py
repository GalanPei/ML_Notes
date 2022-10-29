import torch
from torch import nn
import torch.nn.functional as F

from GNN import aggregator


class GraphSAGE(nn.Module):
    def __init__(self,
                 inputs: torch.Tensor,
                 depth: int,
                 hidden_size: int,
                 neighborhood_fun,
                 activation=F.relu,
                 aggregator_type: str = "mean"):
        """

        :param inputs(:class:`torch.Tensor`):
            Input features of nodes in graph. With shape as (V, hidden_size)
            where V is the number of total nodes
        :param depth(int):
            The depth of graph sampling and aggregating
        :param hidden_size(int):
            The size of weight matrix
        :param neighborhood_fun:
            The neighborhood function for each node. The input is the index of
            one node, and the return value of this function is the list of the
            index of the neighborhood nodes of the input node.
        :param activation:
            Activation function used in this model
        :param aggregator_type(str):
            Aggregator function type, should be 'Mean', 'LSTM' or 'Pooling'
        """
        super(GraphSAGE, self).__init__()
        self.inputs = inputs
        self.K_hidden = [inputs for _ in range(depth)]
        self.V = inputs.shape[0]
        self.depth = depth
        weight = [torch.empty(size=(hidden_size, hidden_size)) for _ in range(self.depth)]
        self.weight = [nn.Parameter(nn.init.xavier_normal_(w)) for w in weight]
        self.activation = activation
        self.neighborhood_fun = neighborhood_fun
        if aggregator_type == "Mean":
            self.aggregator = aggregator.MeanAggregator
        elif aggregator_type == "LSTM":
            self.aggregator = aggregator.LSTMAggregator
        elif aggregator_type == "Pooling":
            self.aggregator = aggregator.PoolingAggregator
        else:
            raise TypeError("Aggregator type cannot be recognized!")

    def forward(self):
        """
        Forward propagation of GraphSAGE for the total graph

        :return:
        """
        Z = self.inputs
        for node in range(self.V):
            Z = self.propagation(node)
        return Z

    def propagation(self, node: int):
        """
        Forward propagation for one single node.

        :param node:
        :return:
        """
        z = self.inputs[node]
        for i in range(1, self.depth + 1):
            z = self.graph_conv(node, i)
        return z

    def graph_conv(self, node: int, k: int):
        """
        Graph convolution operation of one node

        :param node: Index of the given node
        :param k: The layer
        :return new_node_feature: Hidden feature of input node
        """
        neighbour_list = self.neighborhood_fun(node)  # Get the neighborhood nodes
        aggregate = self.aggregator([self.K_hidden[k - 1][i] for i in neighbour_list])
        h_neighbour = aggregate()
        new_node_feature = self.activation(
            self.weight[k] @ torch.concat([self.K_hidden[k - 1][node], h_neighbour]))
        new_node_feature /= torch.norm(new_node_feature, p=2)
        self.K_hidden[k][node] = new_node_feature
        return new_node_feature
