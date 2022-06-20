import torch
import math
import pandas as pd
import numpy as np

# Nadaraya-Watson kernel attention with distance between keys and queries as score
class MLP_NW_dist_att(torch.nn.Module):
    """
    MLP_NW_dist_att only work when number of fetures is identical to the number of target
    """
    def __init__(self,input_size, hidden_size, output_size):
        super(MLP_NW_dist_att, self).__init__()

        # W_keys as an MLP
        self.Wk = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size))

        # W_queries as an MLP
        self.Wq = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size))

    def forward(self,coords_f,values_f,coords_t):

        Wk = self.Wk(coords_f)
        Wq = self.Wq(coords_t)

        # score
        score = torch.cdist(Wk,Wq, p=1)

        # NW attention weight
        self.attention_weights = torch.nn.functional.softmax(-(score)**2 / 2, dim=1)

        # context
        res = torch.einsum('bij,bik->bjk',self.attention_weights,values_f)

        return(res)


# MLP, Nadaraya-Watson kernel attention with distance between keys and queries as score
# MLP of Q,K, V
# FC of attention output
class MLP_NW_dist_2_att(torch.nn.Module):

    def __init__(self,input_k, input_q, input_v, input_t, hidden_size):
        super(MLP_NW_dist_2_att, self).__init__()

        # W_keys as an MLP
        self.Wk = torch.nn.Sequential(
            torch.nn.Linear(input_k, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size))

        # W_queries as an MLP
        self.Wq = torch.nn.Sequential(
            torch.nn.Linear(input_q, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size))

        # W_values as an MLP
        self.Wv = torch.nn.Sequential(
            torch.nn.Linear(input_v, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size))

        # W_ouput
        self.Wo = torch.nn.Linear(hidden_size,input_t)

    def forward(self,coords_f,values_f,coords_t):

        Wk = self.Wk(coords_f) # bij: 64x31x20
        Wq = self.Wq(coords_t) # bkj: 64x10x20
        Wv = self.Wv(values_f) # bij: 64x31x8

        # score
        score = torch.cdist(Wk,Wq, p=1)

        # NW attention weight
        self.attention_weights = torch.nn.functional.softmax(-(score)**2 / 2, dim=1) #bik 64, 31, 10

        # context
        output = torch.einsum('bik,bij->bkj',self.attention_weights,Wv) # => 64 10 20

        res = self.Wo(output)

        return(res)

# Multi-Head Encoder
# as in https://arxiv.org/abs/1706.03762
class multi_head_att(torch.nn.Module):

    def __init__(self,input_k, input_q, input_v, hidden_size, output_size, num_heads=1):
        super(multi_head_att, self).__init__()

        assert hidden_size % num_heads == 0, "Hidden size must be 0 modulo number of heads."

        self.input_k = input_k
        self.input_q = input_q
        self.input_v = input_v
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # W_keys
        self.Wk = torch.nn.Linear(input_k, hidden_size)

        # W_queries
        self.Wq = torch.nn.Linear(input_q, hidden_size)

        # W_values
        self.Wv = torch.nn.Linear(input_v, hidden_size)

        # W_ouput
        self.Wo = torch.nn.Linear(hidden_size,output_size)

    def forward(self,coords_f,values_f,coords_t,return_attention=False):

        Wk = self.Wk(coords_f) # bij: 64x51x8
        Wq = self.Wq(coords_t) # bkj: 64x20x8
        Wv = self.Wv(values_f) # bij: 64x51x8

        ##############
        # multi-head #
        ##############

        # reshaping into a multi-head
        Wk = torch.reshape(Wk,(Wk.shape[0], Wk.shape[1], int(Wk.shape[2]/self.num_heads), self.num_heads)) #bi(j/h)h :64x51x8x3
        Wk = Wk.permute(0,3,1,2) # bhi(j/h)
        Wq = torch.reshape(Wq,(Wq.shape[0], Wq.shape[1], int(Wq.shape[2]/self.num_heads), self.num_heads)) #bk(j/h)h :64x20x8x3
        Wq = Wq.permute(0,3,1,2) # bhk(j/h)
        Wv = torch.reshape(Wv,(Wv.shape[0], Wv.shape[1], int(Wv.shape[2]/self.num_heads), self.num_heads)) #bi(j/h)h :64x51x8x3
        Wv = Wv.permute(0,3,1,2) # bhi(j/h)

        # score: dot prod between Wk and Wq
        d_wk = Wq.shape[-1]
        score = torch.einsum('bhij,bhkj->bhik',Wk,Wq)/math.sqrt(d_wk) #bhik

        # attention weight
        self.attention_weights = torch.nn.functional.softmax(score, dim=1) #bhik

        # context
        output = torch.einsum('bhik,bhij->bhjk',self.attention_weights,Wv) # bh(j/h)k

        #######################
        # Concatenating heads #
        #######################
        output = output.permute(0, 2, 1, 3)  # b(j/h)hk
        output = output.reshape(output.shape[0], output.shape[1]*output.shape[2], output.shape[3]) # bjk
        output = torch.transpose(output,2,1) # bkj

        # prediction
        res = self.Wo(output) # bkl

        if return_attention:
            return res, self.attention_weights
        else:
            return res
