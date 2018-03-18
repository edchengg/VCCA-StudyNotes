import torch
import torch.nn as nn
class Network_reconstruction(nn.Module):

    def __init__(self, input_size, hidden_size, activation=None, dropout=0):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.dropout(x)
        out = self.fc1(out)
        if not self.activation == None:
            out = self.activation(out)
        out_sigma = out
        out_mean = out
        return out_mean, out_sigma