#######                                      ####### 
####### UNDER DEV, might change at any time  #######
#######                                      #######

import torch
from torch import nn
from steams.models.mha import mha1


###
### Transformer AE
###
class Transformer_ae(nn.Module):
    """
    Transformer Auto-encoder
    Args:
        num_layers: nb of enc layers
        input_coords: Input size of the coordinates
        num_heads: Number of heads to use in the attention block
        dim_feedforward: Dimensionality of the hidden layer in the MLP
        dropout: Dropout probability to use in the dropout layers
        linear_hidden_size:  Dimensionnality of the final linear fc
        output_size: Dimensionanlity of the output
    """
    def __init__(self, num_layers=1, input_coords=3, hidden_size=20, num_heads=4, dim_feedforward=20, dropout=0.1, linear_hidden_size=7, output_size=1):
        super().__init__()

        self.encoder = TransformerEncoder(num_layers=num_layers,input_x = input_coords, hidden_size=hidden_size, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.fc = nn.Linear(input_coords, output_size)

    def positioning(self,X,coords):
        X = torch.repeat_interleave(X, coords.size(-1), dim=-1)
        X = X + coords
        return X

    def forward(self, coords,values):
        X = self.positioning(values,coords)
        X_enc = self.encoder(X)
        return self.fc(X_enc)

###
### Transformer coords
###
class Transformer_coords(nn.Module):
    """
    Transformer
    Parameters of the encoder and decoder are identical.
    Args:
        num_layers: nb of enc/dec layers
        input_coords: Input size of the coordinates
        num_heads: Number of heads to use in the attention block
        dim_feedforward: Dimensionality of the hidden layer in the MLP
        dropout: Dropout probability to use in the dropout layers
        linear_hidden_size:  Dimensionnality of the final linear fc
        output_size: Dimensionanlity of the output
    """

    def __init__(self,num_layers=1, input_coords=3, hidden_size=20, num_heads=4, dim_feedforward=20, dropout=0.1, linear_hidden_size=7, output_size=1):
        super().__init__()

        self.encoder = TransformerEncoder(num_layers=num_layers,input_x = input_coords, hidden_size=hidden_size, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.decoder = TransformerDecoder(num_layers=num_layers,input_y = input_coords, hidden_size=hidden_size, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.fc = torch.nn.Linear(input_coords, output_size)

    #def positioning(self,X,coords):
    #    X = torch.repeat_interleave(X, coords.size(-1), dim=-1)
    #    X = X + coords
    #    return X

    def positioning(self,KEY,VALUE):
        X = torch.cat((KEY,VALUE), dim=-1)
        return X

    def forward(self, KEY_X, VALUE_X, KEY_Y, VALUE_Y):

        X = self.positioning(KEY_X, VALUE_X)
        Y = self.positioning(KEY_Y, VALUE_Y)

        X_enc = self.encoder(X)
        Y, X_enc = self.decoder(Y, X_enc)

        output = self.fc(Y)
        return output, X_enc


###
### Transformer Encoder
###
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
        self.self_attn = mha1(self.input_k, self.input_q, self.input_v, hidden_size, self.input_t, num_heads)

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

###
### Transformer Decoder
###
class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, Y,X_enc):
        self.attention_maps = [[] * len(self.layers) for _ in range (2)]
        for layer in self.layers:
            _, attn_map = layer.self_attn1(Y,Y,Y, return_attention=True)
            self.attention_maps[0].append(attn_map)
            _, attn_map = layer.attn2(X_enc,Y,Y, return_attention=True)
            self.attention_maps[1].append(attn_map)
            Y,X_enc = layer(Y, X_enc)

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
        self.self_attn1 = mha1(self.input_k, self.input_q, self.input_v, hidden_size, self.input_t, num_heads)
        # Layers to apply in between the main layers
        self.norm1 = torch.nn.LayerNorm(self.input_t)

        # Attention layer 2
        self.attn2 = mha1(self.input_k, self.input_q, self.input_v,hidden_size, self.input_t, num_heads)

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
        attn_out2 = self.attn2(y,y,X_enc) # rem: (KEY,VALUE,QUERY) with query from X_enc

        # Add and Norm 2
        y2 = self.norm2(y + self.dropout(attn_out2))

        # FFN
        linear_out = self.linear_net(y2)

        # Add and Norm
        y3 = self.norm3(y2 + self.dropout(linear_out))

        return y3, X_enc
