import torch
from torch import nn
import math

from steams.models.attention import mha, madsnn4


class EncoderTransformerBlock(nn.Module):
    def __init__(self, input_size,hidden_size,num_heads,dim_feedforward,dropout=0.1):
        super().__init__()

        # Self-Attention layer
        self.attention = mha(input_size, input_size, input_size, hidden_size, num_heads,dropout)

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
        attn_out = self.attention(X,X,X)

        # Add and Norm (residual)
        res = self.norm1(X + attn_out)

        # FFN
        linear_out = self.ffn(res)

        # Add and Norm (residual)
        res = self.norm2(res + self.dropout(linear_out))

        return res


class EncoderTransformer(nn.Module):
    def __init__(self, input_k, input_q, input_v, hidden_size,num_heads,dim_feedforward,num_blk,dropout=0.1,max_len=5000):
        super().__init__()

        # Self-Attention layer
        input_size = input_k + input_q + input_v

        self.transformer_blocks = nn.ModuleList(
            [EncoderTransformerBlock(input_size, hidden_size,num_heads, dim_feedforward) for _ in range(num_blk)]
        )

    def forward(self, K,V,Q):

        X = torch.cat((K,V,Q),dim=2)
        
        self.attention_weights = [None]*len(self.transformer_blocks)
        for i, blk in enumerate(self.transformer_blocks):
            X = blk(X)
            self.attention_weights[i] = blk.attention.weights
        
        return X
    

class EncoderTransformer2(nn.Module):
    def __init__(self, input_k, input_q, input_v, hidden_size,num_heads,dim_feedforward,num_blk,dropout=0.1):
        super().__init__()

        # Self-Attention layer
        input_size = input_k + input_q + input_v

        self.transformer_blocks = nn.ModuleList(
            [EncoderTransformerBlock(input_size, hidden_size, num_heads,dim_feedforward) for _ in range(num_blk)]
        )

        self.linear = nn.Linear(input_size, input_v)
        

    def forward(self, K,V,Q):

        X = torch.cat((K,V,Q),dim=2)
        self.attention_weights = [None]*len(self.transformer_blocks)
        for i, blk in enumerate(self.transformer_blocks):
            X = blk(X)
            self.attention_weights[i] = blk.attention.weights
        
        
        output = self.linear(X)
        return output
    

class EncoderTransformer3(nn.Module):
    def __init__(self, input_k, input_q, input_v, nb_variable,embedding_dim,hidden_size,num_heads,dim_feedforward,num_blk,dropout=0.1):
        super().__init__()
        
        self.variable_embedding = nn.Embedding(nb_variable, embedding_dim)
        
        self.transformer = EncoderTransformer2(input_k=input_k+embedding_dim, input_q=input_q+embedding_dim, input_v=input_v,hidden_size=hidden_size,num_heads=num_heads,dim_feedforward=dim_feedforward,num_blk=num_blk,dropout=dropout)
        

    def forward(self, K,K_v,V,Q,Q_v):

        K_v_embedded = self.variable_embedding(K_v.int())
        Q_v_embedded = self.variable_embedding(Q_v.int())

        K_v_embedded = torch.reshape(K_v_embedded,(K_v_embedded.shape[0],K_v_embedded.shape[1],K_v_embedded.shape[2]*K_v_embedded.shape[3]))
        Q_v_embedded = torch.reshape(Q_v_embedded,(Q_v_embedded.shape[0],Q_v_embedded.shape[1],Q_v_embedded.shape[2]*Q_v_embedded.shape[3]))

        K_new = torch.cat((K,K_v_embedded),dim=2)
        Q_new= torch.cat((Q,Q_v_embedded),dim=2)

        output = self.transformer(K_new,V,Q_new)
        return output

class Transf_enc_dec(nn.Module):
    def __init__(self, input_k, input_q, input_v, hidden_size,num_heads,dim_feedforward,num_blk,dropout=0.1):
        super().__init__()

        # Self-Attention layer
        input_size = input_k + input_q + input_v

        self.enc_transf = EncoderTransformer(input_k=input_k, input_q=input_q, input_v=input_v, hidden_size=hidden_size,num_heads=num_heads,
                                             dim_feedforward=dim_feedforward,num_blk=num_blk,dropout=dropout)

        # Cross-Attention layer
        self.cross_att = mha(input_k=input_size, input_q=input_q, input_v=input_size, hidden_size=hidden_size, num_heads=num_heads,dropout=dropout)


        self.linear = nn.Linear(input_size, input_v)

    def forward(self, K,V,Q):
        
        enc = self.enc_transf(K,V,Q)

        Y = self.cross_att(enc,enc,Q)

        output = self.linear(Y)

        return output
    

class PositionalEncoding(nn.Module):
    def __init__(self, input_size, max_len=5000):
        super().__init__()
        # Create a fixed sinusoidal positional encoding matrix
        pe = torch.zeros(max_len, input_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_size, 2) * -(math.log(10000.0) / input_size))

        even_indices = torch.arange(0, input_size, 2)  # Even dimensions
        odd_indices = torch.arange(1, input_size, 2)   # Odd dimensions

        div_term_even = torch.exp(even_indices * -(math.log(10000.0) / input_size))
        div_term_odd = torch.exp(odd_indices * -(math.log(10000.0) / input_size))

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term_even)  # Sine for even dimensions
        pe[:, 1::2] = torch.cos(position * div_term_odd)   # Cosine for odd dimensions
        self.register_buffer('pe', pe)

    def forward(self, X):
        # Add positional encodings to the input
        seq_len = X.size(1)  # Get the sequence length (dim=1)
        return X + self.pe[:seq_len, :]


###################
###################

class EncoderTransformerBlock_mads(nn.Module):
    def __init__(self, device,type,kernel,input_size,hidden_size,dim_feedforward,dropout=0.1):
        super().__init__()

        # Self-Attention layer
        self.attention = madsnn4(device,type,kernel,input_size, input_size, input_size, hidden_size, dropout)

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
        attn_out = self.attention(X,X,X)

        # Add and Norm (residual)
        res = self.norm1(X + attn_out)

        # FFN
        linear_out = self.ffn(res)

        # Add and Norm (residual)
        res = self.norm2(res + self.dropout(linear_out))

        return res


class EncoderTransformer2_mads(nn.Module):
    def __init__(self,device,type,kernel, input_k, input_q, input_v, hidden_size,dim_feedforward,num_blk,dropout=0.1):
        super().__init__()

        # Self-Attention layer
        input_size = input_k + input_q + input_v
        
        self.transformer_blocks = nn.ModuleList(
            [EncoderTransformerBlock_mads(device,type,kernel,input_size, hidden_size, dim_feedforward) for _ in range(num_blk)]
        )

        self.linear = nn.Linear(input_size, input_v)
        

    def forward(self, K,V,Q):

        X = torch.cat((K,V,Q),dim=2)

        self.attention_weights = [None]*len(self.transformer_blocks)
        for i, blk in enumerate(self.transformer_blocks):
            X = blk(X)
            self.attention_weights[i] = blk.attention.attention.weights
        
        
        output = self.linear(X)
        return output
 