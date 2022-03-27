import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, num_head, input_dim, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.num_head = num_head
        self.weight_query = nn.Linear()
        self.weight_key = nn.Linear()
        self.weight_value = nn.Linear()
        

    def forward(self, x):
        pass
