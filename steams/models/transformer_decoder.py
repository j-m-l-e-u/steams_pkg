import torch
from torch import nn
from steams.models.attention import mha

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, Y,X_enc):
        self.attention_maps = [[] * len(self.layers) for _ in range (2)]
        for layer in self.layers:
            _, attn_map = layer.self_attn1(Y,Y,Y, return_attention=True)
            self.attention_maps[0].append(attn_map)
            _, attn_map = layer.attn2(X_enc,X_enc,Y, return_attention=True)
            self.attention_maps[1].append(attn_map)
            Y,X_enc = layer(coords_f,values_t,coords_t,enc_coords_f,enc_values_f,enc_coords_t)

        return Y,X_enc

    def get_attention_maps(self):
        return self.attention_maps

class DecoderBlock(nn.Module):
    def __init__(self, input_y, hidden_size, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.input_k = input_y
        self.input_q = input_y
        self.input_v = input_y
        self.input_t = input_y

        # Attention layer 1
        self.self_attn1 = mha(self.input_k, self.input_q, self.input_v, hidden_size, self.input_t, num_heads)
        # Layers to apply in between the main layers
        self.norm1 = torch.nn.LayerNorm(self.input_t)

        # Attention layer 2
        self.attn2 = mha(self.input_k, self.input_q, self.input_v,hidden_size, self.input_t, num_heads)

        # Layers to apply in between the main layers
        self.norm2 = torch.nn.LayerNorm(self.input_t)

        # Two-layer MLP
        self.linear_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_t, dim_feedforward),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_feedforward, self.input_t),
        )

        # Layers to apply in between the main layers
        self.norm3 = torch.nn.LayerNorm(self.input_t)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, Y, X_enc):

        # Attention 1: self attention
        attn_out1 = self.self_attn1(Y,Y,Y)

        # Add and Norm
        y = self.norm1(Y + self.dropout(attn_out1))

        # Attention 2: Cross-attention
        attn_out2 = self.self_attn2(X_enc,X_enc,y) # rem: (KEY,VALUE,QUERY) with query from y

        # Add and Norm 2
        y2 = self.norm2(y + self.dropout(attn_out2))

        # FFN
        linear_out = self.linear_net(y2)

        # Add and Norm
        y3 = self.norm3(y2 + self.dropout(linear_out))

        return y3, X_enc
