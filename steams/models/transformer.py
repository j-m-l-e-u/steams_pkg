import torch
from torch import nn
from steams.models.attention import multi_head_att

class Transformer_ae(nn.Module):
    """
    Transformer Auto-encoder
    Args:
        num_layers: nb of enc layers
        input_k: Dimensionality of the keys: coordinates of features
        input_q: Dimensionality of the queries: coordinates of targets
        input_v: Dimensionality of the values: values of features
        num_heads: Number of heads to use in the attention block
        dim_feedforward: Dimensionality of the hidden layer in the MLP
        dropout: Dropout probability to use in the dropout layers
        linear_hidden_size:  Dimensionnality of the final linear fc
        output_size: Dimensionanlity of the output
    """
    def __init__(self, num_layers=1, input_k=3, input_q=3, input_v=1, hidden_size=20, num_heads=4, dim_feedforward=20, dropout=0.1, linear_hidden_size=7, output_size=1):
        super().__init__()
        self.encoder = TransformerEncoder(num_layers=num_layers,input_k=input_k, input_q=input_q, input_v=input_v, hidden_size=hidden_size, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.fc = nn.Linear(input_k+input_q+input_v, output_size)

    def forward(self, coords_f,values_f,coords_t):
        enc_k,enc_v,enc_q = self.encoder(coords_f,values_f,coords_t)
        enc_output = torch.cat((enc_k,enc_v,enc_q),dim=-1)
        output = self.fc(enc_output)
        return output

class Transformer(nn.Module):
    """
    Transformer
    Parameters of the encoder and decoder are identical.
    Args:
        num_layers: nb of enc/dec layers
        input_k: Dimensionality of the keys: coordinates of features
        input_q: Dimensionality of the queries: coordinates of targets
        input_v: Dimensionality of the values: values of features
        num_heads: Number of heads to use in the attention block
        dim_feedforward: Dimensionality of the hidden layer in the MLP
        dropout: Dropout probability to use in the dropout layers
        linear_hidden_size:  Dimensionnality of the final linear fc
        output_size: Dimensionanlity of the output
    """

    def __init__(self,num_layers=1, input_k=3, input_q=3, input_v=1, hidden_size=20, num_heads=4, dim_feedforward=20, dropout=0.1, linear_hidden_size=7, output_size=1):
        super(Transformer,self).__init__()
        self.encoder = TransformerEncoder(num_layers=num_layers,input_k=input_k, input_q=input_q, input_v=input_v, hidden_size=hidden_size, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.decoder = TransformerDecoder(num_layers=num_layers,input_k=input_k, input_q=input_q, input_v=input_v, hidden_size=hidden_size, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.fc = torch.nn.Linear(linear_hidden_size, output_size)

    def forward(self, enc_coords_f,enc_values_f,enc_coords_t,dec_coords_f,dec_values_t,dec_coords_t):
        enc_k,enc_v,enc_q = self.encoder(enc_coords_f,enc_values_f,enc_coords_t)
        dec_k,dec_v,dec_q = self.decoder(dec_coords_f,dec_values_t,dec_coords_t,enc_k,enc_v,enc_q)
        dec_output = torch.cat((dec_k,dec_v,dec_q),dim=-1)
        output = self.fc(dec_output)
        return output

class EncoderBlock(nn.Module):
    def __init__(self, input_k, input_q, input_v, hidden_size, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.input_k = input_k
        self.input_q = input_q
        self.input_v = input_v
        self.input_t = input_k + input_q + input_v

        # Attention layer
        self.self_attn = multi_head_att(self.input_k, self.input_q, self.input_v, hidden_size, self.input_t, num_heads)

        # add norm 1
        self.norm1 = torch.nn.LayerNorm(self.input_t)

        # Two-layer MLP
        self.linear_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_t, dim_feedforward),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_feedforward, self.input_t),
        )

        # Layers to apply in between the main layers
        self.norm2 = torch.nn.LayerNorm(self.input_t)

        self.dropout = torch.nn.Dropout(dropout)

    def kvq_split(self,x):
        k = x[:,:,range(self.input_k)]
        v = x[:,:,range(self.input_k,(self.input_k+self.input_v))]
        q = x[:,:,range((self.input_k+self.input_v),(self.input_k+self.input_v+self.input_q))]
        return k,v,q

    def forward(self, coords_f,values_f,coords_t):

        # Attention
        attn_out = self.self_attn(coords_f,values_f,coords_t)

        # Add and Norm
        x = self.dropout(attn_out)
        k,v,q = self.kvq_split(x)
        coords_f = coords_f + k
        values_f = values_f + v
        coords_t = coords_t + q
        x = self.norm1(torch.cat((coords_f,values_f,coords_t),dim=-1))

        # FFN
        linear_out = self.linear_net(x)

        # Add and Norm
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        k,v,q = self.kvq_split(x)

        return k,v,q

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, coords_f,values_f,coords_t):
        self.attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(coords_f,values_f,coords_t, return_attention=True)
            self.attention_maps.append(attn_map)
            coords_f,values_f,coords_t = layer(coords_f,values_f,coords_t)
        return coords_f,values_f,coords_t

    @torch.no_grad()
    def get_attention_maps(self):
        return self.attention_maps


class DecoderBlock(nn.Module):
    def __init__(self, input_k, input_q, input_v, hidden_size, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.input_k = input_k
        self.input_q = input_q
        self.input_v = input_v
        self.input_t = input_k + input_q + input_v

        # Attention layer 1
        self.self_attn1 = multi_head_att(self.input_k, self.input_q, self.input_v, hidden_size, self.input_t, num_heads)
        # Layers to apply in between the main layers
        self.norm1 = torch.nn.LayerNorm(self.input_t)

        # Attention layer 2
        self.self_attn2 = multi_head_att(self.input_k, self.input_q, self.input_v,hidden_size, self.input_t, num_heads)
        # Layers to apply in between the main layers
        self.norm2 = torch.nn.LayerNorm(self.input_t)

        # Two-layer MLP
        self.linear_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_t, dim_feedforward),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_feedforward, self.input_t),
        )

        self.norm3 = torch.nn.LayerNorm(self.input_t)
        self.dropout = torch.nn.Dropout(dropout)

    def kvq_split(self,x):
        k = x[:,:,range(self.input_k)]
        v = x[:,:,range(self.input_k,(self.input_k+self.input_v))]
        q = x[:,:,range((self.input_k+self.input_v),(self.input_k+self.input_v+self.input_q))]
        return k,v,q

    def forward(self, coords_f,values_t,coords_t, enc_coords_f,enc_values_f,enc_coords_t):

        # Attention 1: self attention
        attn_out1 = self.self_attn1(coords_f,values_t,coords_t)

        # Add and Norm 1
        x = self.dropout(attn_out1)
        k,v,q = self.kvq_split(x)
        coords_f = coords_f + k
        values_t = values_t + v
        coords_t = coords_t + q
        y = self.norm1(torch.cat((coords_f,values_t,coords_t),dim=-1))

        # Attention 2: encoder-decoder
        k,v,q = self.kvq_split(y)
        attn_out2 = self.self_attn2(enc_coords_f,enc_values_f,q) # q from y

        # Add and Norm 2
        x2 = self.dropout(attn_out2)
        k2,v2,q2 = self.kvq_split(x2)
        enc_coords_f = enc_coords_f + k2
        enc_values_f = enc_values_f + v2
        q = q + q2
        y2 = self.norm2(torch.cat((enc_coords_f,enc_values_f,q),dim=-1))

        # FFN
        linear_out = self.linear_net(y2)

        # Add and Norm
        y2 = y2 + self.dropout(linear_out)
        y2 = self.norm2(y2)

        k,v,q = self.kvq_split(y2)

        return k,v,q

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, coords_f,values_t,coords_t,enc_coords_f,enc_values_f,enc_coords_t):
        self.attention_maps = [[] * len(self.layers) for _ in range (2)]
        for layer in self.layers:
            _, attn_map = layer.self_attn1(coords_f,values_t,coords_t, return_attention=True)
            self.attention_maps[0].append(attn_map)
            _, attn_map = layer.self_attn2(coords_f,values_t,coords_t, return_attention=True)
            self.attention_maps[1].append(attn_map)
            coords_f,values_t,coords_t = layer(coords_f,values_t,coords_t,enc_coords_f,enc_values_f,enc_coords_t)

        return coords_f,values_t,coords_t

    @torch.no_grad()
    def get_attention_maps(self):
        return self.attention_maps

class TransformerEncDec(nn.Module):
    def __init__(self, encoder, decoder, hidden_size, output_size):
        super(TransformerEncDec, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, enc_coords_f,enc_values_f,enc_coords_t,dec_coords_f,dec_values_t,dec_coords_t):
        enc_k,enc_v,enc_q = self.encoder(enc_coords_f,enc_values_f,enc_coords_t)
        dec_k,dec_v,dec_q = self.decoder(dec_coords_f,dec_values_t,dec_coords_t,enc_k,enc_v,enc_q)
        dec_output = torch.cat((dec_k,dec_v,dec_q),dim=-1)
        output = self.fc(dec_output)
        return output
