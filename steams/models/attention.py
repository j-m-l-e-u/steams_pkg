import torch
from torch import nn
import math

from steams.models.mads import dpnn4


class saBlock(nn.Module):
    def __init__(self, input_size,hidden_size,dim_feedforward,dropout=0.1):
        super().__init__()

        # Self-Attention layer
        self.self_attn = dpnn4(input_size, input_size, input_size, hidden_size, dropout)

        # add norm 1
        self.norm1 = nn.LayerNorm(input_size)

        # Two-layer MLP
        self.ffn = nn.Sequential(
            nn.Linear(input_size, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, input_size),
        )

        # Layers to apply in between the main layers
        self.norm2 = nn.LayerNorm(input_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, X):

        # Attention
        attn_out = self.self_attn(X,X,X)

        # Add and Norm (residual)
        res = self.norm1(X + attn_out)

        # FFN
        linear_out = self.ffn(res)

        # Add and Norm (residual)
        res = self.norm2(res + self.dropout(linear_out))

        return res


class EncoderBlock(nn.Module):
    def __init__(self, input_k, input_q, input_v, hidden_size,dim_feedforward,dropout=0.1):
        super().__init__()

        # Self-Attention layer
        input_size = input_k + input_q + input_v
        self.sab = saBlock(input_size,hidden_size,dim_feedforward,dropout)

    def forward(self, K,V,Q):

        X = torch.cat((K,V,Q),dim=2)

        # Attention
        out = self.sab(X)
        
        return out
