import torch
from torch import nn
from steams.models.attention import mha

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, X):
        self.attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(X,X,X, return_attention=True)
            self.attention_maps.append(attn_map)
            X = layer(X)
        return X

    def get_attention_maps(self):
        return self.attention_maps

class EncoderBlock(nn.Module):
    def __init__(self, input_x, hidden_size, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.input_k = input_x
        self.input_q = input_x
        self.input_v = input_x
        self.input_t = input_x

        # Attention layer
        self.self_attn = mha(self.input_k, self.input_q, self.input_v, hidden_size, self.input_t, num_heads)

        # add norm 1
        self.norm1 = torch.nn.LayerNorm(self.input_t)

        # Two-layer MLP
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(self.input_t, dim_feedforward),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_feedforward, self.input_t),
        )

        # Layers to apply in between the main layers
        self.norm2 = torch.nn.LayerNorm(self.input_t)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, X):

        # Attention
        attn_out = self.self_attn(X,X,X)

        # Add and Norm (residual)
        res = self.norm1(X + self.dropout(attn_out))

        # FFN
        linear_out = self.ffn(res)

        # Add and Norm (residual)
        res = self.norm2(res + self.dropout(linear_out))

        return res
