import torch.nn as nn
import torch
import numpy as np
import torch.functional as f
from multiHeadAttention import *
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, d_model, hidden_size, num_heads, mask=None, dropout=0.1, activation=nn.ReLU()):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mask = mask
        self.dropout = dropout
        self.activation = activation
        self.attention = MultiHeadAttention(d_model, num_heads, hidden_size, mask, dropout)
        self.attention_norm = nn.LayerNorm(d_model)

        feed_forward = [nn.Linear(d_model, hidden_size
                                  ), self.activation, nn.Linear(d_model, hidden_size)]

        if dropout is not None:
            self.attn_droput = nn.Dropout(dropout)
            feed_forward.append(nn.Dropout(dropout))

        self.feed_forward = nn.Sequential(*feed_forward)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, k,q,v):
        # print('attnetion inout size',x.size())
        attn = self.attention(k,q,v)
        add_ = attn + k
        add_dropout = self.attn_droput(add_)
        add_norm = self.attention_norm(add_dropout)
        feedForward = self.feed_forward(add_norm)
        ff_norm = self.norm(feedForward + k)

        return ff_norm
