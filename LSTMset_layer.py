import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import math

class LSTMsetCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_LSTMs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_LSTMs = num_LSTMs
        self.W = nn.Parameter(torch.Tensor(num_LSTMs, input_size, hidden_size * 4))
        self.U = nn.Parameter(torch.Tensor(num_LSTMs, hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(num_LSTMs, 1, hidden_size * 4))
        self.bias_not_used = nn.Parameter(torch.Tensor(num_LSTMs, 1, hidden_size * 4)) #add this to match pytorch implementation
        self.init_weights()
                
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
         
    def forward(self, x, 
                hidden_states=None):
        """ Inputs: x, (hidden_state, cell_state) 
        x: (batch, # of LSTMs, input_size)
        hidden_state: (batch, # of LSTMs, hidden_size)
        cell_state: (batch, # of LSTMs, hidden_size)
        Outputs:
        hidden_state: (batch, # of LSTMs, hidden_size)
        cell_state: (batch, # of LSTMs, hidden_size)
        """
        n_batch, n_LSTM, _ = x.size()

        if hidden_states is None:
            h_t, c_t = (torch.zeros(n_batch, n_LSTM, self.hidden_size).to(x.device), 
                        torch.zeros(n_batch, n_LSTM, self.hidden_size).to(x.device))
        else:
            h_t, c_t = hidden_states
        x_t = x.permute(1, 0, 2)
        h_t = h_t.permute(1, 0, 2)
        c_t = c_t.permute(1, 0, 2)
        # parallel the computation
        gates = torch.bmm(x_t,self.W) + torch.bmm(h_t,self.U) + self.bias
        i_t, f_t, g_t, o_t = (
            torch.sigmoid(gates[:, :, :self.hidden_size]), # input
            torch.sigmoid(gates[:, :, self.hidden_size:self.hidden_size*2]), # forget
            torch.tanh(gates[:, :, self.hidden_size*2:self.hidden_size*3]),
            torch.sigmoid(gates[:, :, self.hidden_size*3:]), # output
        )
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)        
        return h_t.permute(1, 0, 2), c_t.permute(1, 0, 2)

class LSTMsetLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_LSTMs):
        super().__init__()
        self.LSTMsetCell = LSTMsetCell(input_size, hidden_size, num_LSTMs)
         
    def forward(self, inputs, 
                hidden_states=None):
        """ Inputs: inputs, (hidden_state, cell_state) 
        inputs: (batch, time, # of LSTMs, input_size)
        hidden_state: (batch, # of LSTMs, hidden_size)
        cell_state: (batch, # of LSTMs, hidden_size)
        Outputs: (batch, time, # of LSTMs, hidden_size)
        """
        n_batch, _, n_LSTM, _ = inputs.size()
        output_list=[]
        for i,x in enumerate(torch.unbind(inputs, dim=1)):
            hidden_states = self.LSTMsetCell(x,hidden_states)
            output_list.append(hidden_states[0][:,None,:,:])
        return torch.cat(output_list,dim=1)
