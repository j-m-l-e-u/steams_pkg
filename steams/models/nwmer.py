#######
####### UNDER DEV, might change at any time
#######

import torch
from torch import nn
from steams.models.attention import NW0


###
### NW0mer
###
class NW0mer(nn.Module):
    """
    NW0mer
    """
    def __init__(self,num_layers=1, input_k=3, input_v=1,hidden_size=20, dim_feedforward=20, dropout=0.1):
        super().__init__()

        input_size = input_k + input_v
        #input_size = input_k

        self.encoder = NW0merEncoder(num_layers=num_layers,input_size=input_size, hidden_size=hidden_size, dim_feedforward=dim_feedforward, dropout=dropout)
        self.decoder = NW0merDecoder(num_layers=num_layers,input_size=input_size, hidden_size=hidden_size, dim_feedforward=dim_feedforward, dropout=dropout)
        self.fc = nn.Linear(input_size, input_v)

    def positioning(self,KEY,VALUE):
        X = torch.cat((KEY,VALUE), dim=-1)
        return X
    #def positioning(self,KEY,VALUE):
    #    X = torch.repeat_interleave(VALUE, KEY.size(-1), dim=-1)
    #    X = X + KEY
    #    return X


    def forward(self, QUERY_X, VALUE_X, KEY_Y, VALUE_Y):

        X = self.positioning(QUERY_X, VALUE_X)
        Y = self.positioning(KEY_Y, VALUE_Y)

        X_enc = self.encoder(X)
        Y, X_enc = self.decoder(Y, X_enc)

        output = self.fc(Y)
        return output, X_enc


###
### NWormer AE
###
class NW0mer_ae(nn.Module):
    def __init__(self, num_layers=1, input_k=3, input_v=1, hidden_size=20, dim_feedforward=20, dropout=0.1):
        super().__init__()
        input_size = input_k + input_v

        self.encoder = NW0merEncoder(num_layers=num_layers,input_size=input_size, hidden_size=hidden_size, dim_feedforward=dim_feedforward, dropout=dropout)
        self.fc = nn.Linear(input_size, input_v)

    def positioning(self,KEY,VALUE):
        X = torch.cat((KEY,VALUE), dim=-1)
        return X

    def forward(self, KEY,VALUE):
        X = self.positioning(KEY,VALUE)
        res = self.encoder(X)
        return self.fc(res)


###
### NW0mer Encoder
###
class NW0merEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([NW0EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, X):
        self.attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(X,X,X, return_attention=True) # QUERY = KEY
            self.attention_maps.append(attn_map)
            X = layer(X)
        return X

    def get_attention_maps(self):
        return self.attention_maps

class NW0EncoderBlock(nn.Module):
    def __init__(self, input_size,hidden_size,dim_feedforward,dropout=0.1):
        super().__init__()

        # Self-Attention layer
        self.self_attn = NW0(input_size, hidden_size, dropout)

        # add norm 1
        self.norm1 = torch.nn.LayerNorm(input_size)

        # Two-layer MLP
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(input_size, dim_feedforward),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_feedforward, input_size),
        )

        # Layers to apply in between the main layers
        self.norm2 = torch.nn.LayerNorm(input_size)

        self.dropout = torch.nn.Dropout(dropout)

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


###
### NW0mer Decoder
###
class NW0merDecoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([NW0DecoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, Y,X_enc):
        self.attention_maps = [[] * len(self.layers) for _ in range (2)]
        for layer in self.layers:
            _, attn_map = layer.self_attn(Y,Y,Y, return_attention=True)
            self.attention_maps[0].append(attn_map)
            _, attn_map = layer.cross_attn(X_enc,Y,Y, return_attention=True)
            self.attention_maps[1].append(attn_map)
            Y,X_enc = layer(Y, X_enc)

        return Y,X_enc

    def get_attention_maps(self):
        return self.attention_maps

class NW0DecoderBlock(nn.Module):
    def __init__(self, input_size,hidden_size,dim_feedforward,dropout=0.1):
        super().__init__()

        # Self-Attention layer
        self.self_attn = NW0(input_size, hidden_size, dropout)

        # Layers to apply in between the main layers
        self.norm1 = torch.nn.LayerNorm(input_size)

        # Cross-Attention
        self.cross_attn = NW0(input_size, hidden_size, dropout)

        # Layers to apply in between the main layers
        self.norm2 = torch.nn.LayerNorm(input_size)

        # FFN
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(input_size, dim_feedforward),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_feedforward, input_size),
        )

        # Layers to apply in between the main layers
        self.norm3 = torch.nn.LayerNorm(input_size)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, Y, X_enc):

        # Attention 1: self attention
        attn_out1 = self.self_attn(Y,Y,Y)

        # Add and Norm
        y = self.norm1(Y + self.dropout(attn_out1))

        # Attention 2: Cross-attention
        attn_out2 = self.cross_attn(y,y,X_enc) # rem: (KEY,VALUE,QUERY) with query from X_enc

        # Add and Norm 2
        y2 = self.norm2(y + self.dropout(attn_out2))

        # FFN
        linear_out = self.ffn(y2)

        # Add and Norm
        y3 = self.norm3(y2 + self.dropout(linear_out))

        return y3, X_enc
