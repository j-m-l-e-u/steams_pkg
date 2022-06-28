import torch
from torch import nn
from steams.models.transformer_encoder import TransformerEncoder
from steams.models.transformer_decoder import TransformerDecoder

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

class Transformer(nn.Module):
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
        super(Transformer,self).__init__()

        self.encoder = TransformerEncoder(num_layers=num_layers,input_x = input_coords, hidden_size=hidden_size, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.decoder = TransformerDecoder(num_layers=num_layers,input_x = input_coords, hidden_size=hidden_size, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.fc = torch.nn.Linear(linear_hidden_size, output_size)

    def positioning(self,X,coords):
        X = torch.repeat_interleave(X, coords.size(-1), dim=-1)
        X = X + coords
        return X

    def forward(self, values_X,coords_X,values_Y,coords_Y):

        X = self.positioning(values_X,coords_X)
        Y = self.positioning(values_Y,coords_Y)

        X_enc = self.encoder(X,X,X)
        Y, X_enc = self.decoder(Y, X_enc)

        output = self.fc(Y)
        return output, X_enc
