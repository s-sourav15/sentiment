import torch
import math
import torch.nn as nn
import numpy as np
class Position(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space.
        self.pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        # print('pe intermediate',self.pe.size())


        self.pe = self.pe.unsqueeze(0)

        # self.register_buffer('pe', pe)


    def forward(self, input,position):

        length = input.size(position)

        return self.pe[:, :length]

def get_pos_onehot(length):
    onehot = torch.zeros(length,length)
    idxs = torch.arange(length).long().view(-1,1)
    onehot.scatter_(1,idxs,1)
    return onehot

if __name__ == "__main__":
    print(Position(1))
