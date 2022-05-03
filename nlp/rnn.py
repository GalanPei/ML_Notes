import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


def initialize_rnn_params(input_size: int, hidden_size: int):
    """
    Initialize the parameters of RNN with Gaussian distribution

    :param input_size:
    :param hidden_size: Dimension of the hidden states
    :return: params = [W_xh, W_hh, b_h, W_hq, b_q]
    """
    output_size = input_size
    W_xh = torch.randn((input_size, hidden_size)) * .1
    W_hh = torch.randn((hidden_size, hidden_size)) * .1
    b_h = torch.zeros(hidden_size)
    W_hq = torch.randn((hidden_size, output_size)) * .1
    b_q = torch.zeros(output_size)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def initialize_LSTM_params(input_size: int, hidden_size: int):
    """

    :param input_size:
    :param hidden_size:
    :return:
    """
    output_size = input_size

    def init_gate():
        return [torch.randn((input_size, hidden_size)) * .1,
                torch.randn((hidden_size, hidden_size)) * .1,
                torch.zeros(hidden_size)]

    W_xf, W_hf, b_f = init_gate()  # Forget gate
    W_xi, W_hi, b_i = init_gate()  # Input gate
    W_xc, W_hc, b_c = init_gate()  # Cell gate
    W_xo, W_ho, b_o = init_gate()  # Output gate
    W_hq, b_q = torch.randn((hidden_size, output_size)) * .1, torch.zeros(output_size)

    params = [W_xf, W_hf, b_f,
              W_xi, W_hi, b_i,
              W_xc, W_hc, b_c,
              W_xo, W_ho, b_o,
              W_hq, b_q]
    for param in params:
        param.requires_grad_(True)


def init_rnn_state(batch_size: int, hidden_size: int):
    """

    :param batch_size:
    :param hidden_size:
    :return:
    """
    return torch.zeros((batch_size, hidden_size))


def rnn(inputs, state, params):
    """

    :param inputs:
    :param state:
    :param params:
    :return:
    """
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []

    for x in inputs:
        H = torch.tanh(x @ W_xh + H @ W_hh + b_h)
        Y = H @ W_hq + b_q
        outputs.append(Y)

    return torch.cat(outputs, dim=0), (H,)


def LSTM(inputs, state, params):
    """

    :param inputs:
    :param state:
    :param params:
    :return:
    """
    [W_xf, W_hf, b_f,
     W_xi, W_hi, b_i,
     W_xc, W_hc, b_c,
     W_xo, W_ho, b_o,
     W_hq, b_q] = params
    h, c = state
    outputs = []

    for x in inputs:
        i_t = torch.sigmoid((W_xi @ x) + (W_hi @ h) + b_i)
        f_t = torch.sigmoid((W_xf @ x) + (W_hf @ h) + b_f)
        tilde_c = torch.tanh((W_xc @ x) + (W_hc @ h) + b_c)
        o_t = torch.sigmoid((W_xo @ x) + (W_ho @ h) + b_o)
        c_t = f_t * c + i_t * tilde_c
        h_t = o_t * torch.tanh(c_t)
        Y = W_hq @ h_t + b_q
        outputs.append(Y)

    return torch.cat(outputs, dim=0), (h_t, c_t)




class RNNModelScratch(object):
    def __init__(self, vocab_size,
                 hidden_size,
                 get_params,
                 init_state,
                 forward_func):
        self.input_size = vocab_size

    def __call__(self, X):
        pass

    def begin_state(self):
        pass


def grad_clipping(net, theta):
    """
    Do gradient clipping: $g <- min(1, theta / ||g||) * g)$
    :param net:
    :param theta:
    :return:
    """
    if isinstance(net, nn.Module):
        # params = [param ]
        pass
