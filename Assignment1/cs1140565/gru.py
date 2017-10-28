import math
import torch
from torch.nn import Parameter
from torch.nn.modules.rnn import RNNCellBase
import torch.nn as nn


class CustomGRUCell(RNNCellBase):
    """A custom gated recurrent unit (GRU) cell
    .. math::
        \begin{array}{ll}
        r = \mathrm{sigmoid}(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \mathrm{sigmoid}(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h' = (1 - z) * n + z * h
        \end{array}
    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If `False`, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: `True`
    Inputs: input, hidden
        - **input** (batch, input_size): tensor containing input features
        - **hidden** (batch, hidden_size): tensor containing the initial hidden
          state for each element in the batch.
    Outputs: h'
        - **h'**: (batch, hidden_size): tensor containing the next hidden state
          for each element in the batch
    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(3*hidden_size x input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(3*hidden_size x hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(3*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(3*hidden_size)`
    Examples::
        >>> from gru import CustomGRUCell
        >>> rnn = CustomGRUCell(10, 20)
        >>> input = Variable(torch.randn(6, 3, 10))
        >>> hx = Variable(torch.randn(3, 20))
        >>> output = []
        >>> for i in range(6):
        ...     hx = rnn(input[i], hx)
        ...     output.append(hx)
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(CustomGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, inp, hidden):
        # YOUR CODE HERE
        #input->hidden->
        w_r1 = self.weight_ih[0:self.hidden_size,:]
        w_r2 = self.weight_hh[0:self.hidden_size,:]
        w_u1 =self.weight_ih[self.hidden_size:2*self.hidden_size,:]
        w_u2  = self.weight_hh[self.hidden_size:2*self.hidden_size,:]
        w_o1 = self.weight_ih[2*self.hidden_size:3*self.hidden_size,:]
        w_o2 = self.weight_hh[2*self.hidden_size:3*self.hidden_size,:]
        if(self.bias):
            b_r1 = self.bias_ih[0:self.hidden_size,:]
            b_r2 = self.bias_hh[0:self.hidden_size,:]
            b_u1 = self.bias_ih[self.hidden_size:2*self.hidden_size,:]
            b_u2 = self.bias_hh[self.hidden_size:2*self.hidden_size,:]
            b_o1 = self.bias_ih[2*self.hidden_size:3*self.hidden_size,:]
            b_o2 = self.bias_hh[2*self.hidden_size:3*self.hidden_size,:]
        if(self.bias):
            reset_gate = nn.Sigmoid( torch.mm(w_r1,inp) + torch.mm(w_r2,hidden) + b_r1 + b_r2 )
            update_gate = nn.Sigmoid(torch.mm(w_u1,inp) + torch.mm(w_u2,hidden) + b_u1 + b_u2)
            output = nn.Tanh( torch.mm(w_o1,inp) + torch.cmul(torch.mm(w_o2,hidden), reset_gate) +  b_o1 + b_o2 )
            ans = torch.cmul( (1-update_gate),output) + torch.cmul(update_gate,hidden)
        else:
            reset_gate = nn.Sigmoid( torch.mm(w_r1,inp) + torch.mm(w_r2,hidden))
            update_gate = nn.Sigmoid(torch.mm(w_u1,inp) + torch.mm(w_u2,hidden) )
            output = nn.Tanh( torch.mm(w_o1,inp) + torch.cmul(torch.mm(w_o2,hidden), reset_gate))
            ans = torch.cmul( (1-update_gate),output) + torch.cmul(update_gate,hidden)
        return ans
        