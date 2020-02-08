import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as f
import math


def attention(k, q, v, d_k, mask=None, dropout=0.1):
    mult = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(d_k)
    weights = f.softmax(mult)
    if mask is not None:
        weights += mask.unsqueeze(0)
    # value=torch.matmul(v,weights)
    weights = f.softmax(weights / torch.sqrt(torch.Tensor([d_k * 1.0]).to(weights)))
    value = torch.matmul(weights, v)
    value = value.transpose(1, 2).contiguous()
    return value, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, hidden_size, mask=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.mask = mask
        self.q_linear = nn.Linear(d_model, hidden_size)
        self.k_linear = nn.Linear(d_model, hidden_size)
        self.v_linear = nn.Linear(d_model, hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(self, k, q, v):
        batch_size = q.size(0)
        hidden_size = q.size(1)
        q_proj = self.q_linear(q).view(batch_size, hidden_size, self.num_heads, self.d_k).transpose(1, 2)
        k_proj = self.k_linear(k).view(batch_size, hidden_size, self.num_heads, self.d_k).transpose(1, 2)
        v_proj = self.v_linear(v).view(batch_size, hidden_size, self.num_heads, self.d_k).transpose(1, 2)

        attn_output, weights = attention(k_proj, q_proj, v_proj, self.d_k, self.mask, self.dropout)

        # output=self.output(attn_output)
        output = self.output(attn_output.view(q.size(0), q.size(1), self.hidden_size))
        self.weights = weights

        return output
