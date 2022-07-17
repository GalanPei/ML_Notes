import torch
from torch import nn
from torch.nn import functional as F


def plot_hotmaps(attention_weight):
    """
    Visualize the weight of attention
    """
    

def masked_softmax(input, valid_lens):
    if valid_lens is None:
    # Do normal softmax operation when not given any param
        return F.softmax(input, dim=-1)
    
    