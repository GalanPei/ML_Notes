import torch
from torch import nn
import torch.nn.functional as F

from GNN import aggregator


class GraphSAGE(nn.Module):
    def __init__(self,
                 inputs: torch.Tensor,
                 depth: int,
                 weight: torch.Tensor,
                 neighborhood_fun,
                 activation=F.relu,
                 aggregator_type: str = "mean"):
        """

        :param inputs(:class:`torch.Tensor`):
            Input features of nodes in graph. With shape as (V, hidden_size)
            where V is the number of total nodes
        :param depth(int):
            The depth of graph sampling and aggregating
        :param weight(:class:`torch.Tensor`):
            Weight matrix for each layer
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
        self.V = inputs.shape[0]
        self.depth = depth
        self.weight = weight
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
        Forward propagation of GraphSAGE

        :return:
        """
        for idx in range(self.V):
            h0 = self.inputs[idx]
            for i in range(self.depth):
                for j in range(self.V):
                    pass

    def graph_conv(self, input_node: int, node_features: torch.Tensor, k: int):
        """
        Graph convolution operation of one node

        :param input_node: Index of
        :param node_features:
        :param k:
        :return:
        """
        neighbour_node = self.neighborhood_fun(input_node)
        h_neighbour = self.aggregator([node_features[i] for i in neighbour_node])
        new_node_feature = self.activation(
            self.weight[k] @ torch.concat([node_features[input_node], h_neighbour]))
        return new_node_feature
