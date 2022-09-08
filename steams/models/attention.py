import torch
import math
import pandas as pd
import numpy as np

##############################################
### Nadaraya-Watson kernel family attention  #
##############################################

#######
####### Distance attention
######
class NW0(torch.nn.Module):
    """
    Score is based on the distance of scaled coordinates.
    Weights scale the coordinates and are similar to non-uniform ranges for the variogram.
    We assume VALUE_X and VALUE_Y belonging to one identic process.
    So the scaling of KEY and QUERY is applied with an identic weight `W`.
    Attention follows NW kernel with Gaussian Kernel.(sim. Gaussian variogram).
    """
    def __init__(self, input_size,hidden_size,dropout=0.1):
        super(NW0, self).__init__()

        self.W = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, input_size))

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, KEY,VALUE,QUERY,return_attention=False):

        Wk = self.W(KEY)
        Wq = self.W(QUERY)

        # scaling
        KEY_scale = torch.einsum('bij,bij->bij',KEY , Wk)
        QUERY_scale = torch.einsum('bij,bij->bij',QUERY , Wq)

        # score
        score = torch.cdist(KEY_scale,QUERY_scale, p=2) # here, p=2

        self.attention_weights = torch.nn.functional.softmax(-torch.pow(score,2) / 2, dim=1)

        # context
        context = torch.einsum('bij,bik->bjk',self.attention_weights,VALUE)

        #output
        res = self.dropout(context)

        if return_attention:
            return res, self.attention_weights
        else:
            return res


class NW1(torch.nn.Module):
    """
    Score is based on the distance of `Wk` and `Wq`.
    Weights scale the coordinates and are similar to non-uniform ranges for the variogram.
    We assume VALUE_X and VALUE_Y not belonging to one identic process.
    Attention follows NW kernel with sqrtGaussian Kernel (sim. exponential variogram).
    """
    def __init__(self, input_size,hidden_size,dropout=0.1):
        super(NW1, self).__init__()

        self.W = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, input_size))

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, KEY,VALUE,QUERY,return_attention=False):

        Wk = self.W(KEY)
        Wq = self.W(QUERY)

        # scaling
        KEY_scale = torch.einsum('bij,bij->bij',KEY , Wk)
        QUERY_scale = torch.einsum('bij,bij->bij',QUERY , Wq)

        # score
        score = torch.cdist(Wk,Wq, p=2) # here, p=2

        self.attention_weights = torch.nn.functional.softmax(-(score) / 2, dim=1)

        # context
        context = torch.einsum('bij,bik->bjk',self.attention_weights,VALUE)

        #output
        res = self.dropout(context)

        if return_attention:
            return res, self.attention_weights
        else:
            return res


class NW2(torch.nn.Module):

    def __init__(self,input_k, input_q, input_v, hidden_size,dropout=0.1):
        super(NW2, self).__init__()

        # W_keys as an MLP
        self.W = torch.nn.Sequential(
            torch.nn.Linear(input_k, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, input_k),
            )

        # W_values as an MLP
        self.Wo = torch.nn.Sequential(
            torch.nn.Linear(input_q, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, input_v),
            )

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self,KEY,VALUE,QUERY,return_attention=False):

        Wk = self.W(KEY)
        Wq = self.W(QUERY)
        Wo = self.Wo(QUERY)  # with QUERY

        # scaling
        KEY_scale = torch.einsum('bij,bij->bij',KEY , Wk)
        QUERY_scale = torch.einsum('bij,bij->bij',QUERY , Wq)

        # score
        score = torch.cdist(KEY_scale,QUERY_scale, p=1)

        # attention
        self.attention_weights = torch.nn.functional.softmax(-torch.pow(score,2) / 2, dim=1)

        # context
        context = torch.einsum('bik,bij->bkj',self.attention_weights,VALUE)

        context_scale = torch.einsum('bij,bij->bij',context , Wo)

        # output
        res = self.dropout(context_scale)

        if return_attention:
            return res, self.attention_weights
        else:
            return res

class NW3(torch.nn.Module):
    """
    Score is based on the distance of scaled coordinates.
    Weights scale the coordinates and are similar to non-uniform ranges for the variogram.
    We assume VALUE_X and VALUE_Y not belonging to one identic process.
    So the scaling of KEY and QUERY is applied with two different weight `Wk` and `Wq`.
    Besides, W0 convert VALUE_X into VALUE_Y.
    Attention follows NW kernel with a modified Gaussian Kernel.
    """
    def __init__(self, input_k,input_v,hidden_size,dropout=0.1):
        super(NW3, self).__init__()

        self.Wk = torch.nn.Sequential(
            torch.nn.Linear(input_k, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, input_k))

        self.Wq = torch.nn.Sequential(
            torch.nn.Linear(input_k, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, input_k))

        self.Wo = torch.nn.Sequential(
            torch.nn.Linear(input_v, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, input_v))

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, KEY,VALUE,QUERY,return_attention=False):

        Wk = self.Wk(KEY)
        Wq = self.Wq(QUERY)

        # scaling
        KEY_scale = torch.einsum('bij,bij->bij',KEY , Wk)
        QUERY_scale = torch.einsum('bij,bij->bij',QUERY , Wq)

        # score
        score = torch.cdist(KEY_scale,QUERY_scale, p=1)

        # attention
        self.attention_weights = torch.nn.functional.softmax(-torch.pow(score,2) / 2, dim=1)

        # context
        context = torch.einsum('bij,bik->bjk',self.attention_weights,VALUE)

        # output
        res = self.dropout(self.Wo(context))

        if return_attention:
            return res, self.attention_weights
        else:
            return res

########
######## Scaled-dot-prod attention
########

class NW4(torch.nn.Module):
    """
    Score is based on scaled dot product of scaled coordinates.
    Weights scale the coordinates and are similar to non-uniform ranges for the variogram.
    We assume VALUE_X and VALUE_Y belonging to one identic process.
    So the scaling of KEY and QUERY is applied with one identical weight `W`.
    Attention follows NW kernel with Gaussian Kernel.
    """
    def __init__(self, input_size,input_v, input_t, hidden_size,dropout=0.1):
        super(NW4, self).__init__()

        self.W = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, input_size))

        self.Wv = torch.nn.Sequential(
            torch.nn.Linear(input_v, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size))

        # W_ouput
        self.Wo = torch.nn.Linear(hidden_size,input_t)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, KEY,VALUE,QUERY,return_attention=False):

        Wk = self.W(KEY)
        Wq = self.W(QUERY)
        Wv = self.Wv(VALUE)

        # scaling
        KEY_scale = torch.einsum('bij,bij->bij',KEY , Wk)
        QUERY_scale = torch.einsum('bij,bij->bij',QUERY , Wq)
        VALUE_scale = torch.einsum('bij,bij->bij',VALUE , Wv)

        # score: dot prod between Wk and Wq
        d_wk = KEY_scale.shape[-1]
        score = torch.einsum('bij,bkj->bik',KEY_scale,QUERY_scale)/math.sqrt(d_wk)

        # attention
        self.attention_weights = torch.nn.functional.softmax(-torch.pow(score,2) / 2, dim=1)

        # context
        context = torch.einsum('bik,bij->bkj',self.attention_weights,VALUE_scale) # => 64 10 20

        # output
        res = self.dropout(self.Wo(context))

        if return_attention:
            return res, self.attention_weights
        else:
            return res


class NW5(torch.nn.Module):
    """
    NW w/ scaled dot product
    """
    def __init__(self,input_k, input_q, input_v, input_t, hidden_size, dropout=0.1):
        super(NW5, self).__init__()

        # W_keys as an MLP
        self.Wk = torch.nn.Sequential(
            torch.nn.Linear(input_k, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, input_k),
            )

        # W_queries as an MLP
        self.Wq = torch.nn.Sequential(
            torch.nn.Linear(input_q, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, input_q),
            )

        # W_values as an MLP
        self.Wv = torch.nn.Sequential(
            torch.nn.Linear(input_v, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            )

        # W_ouput
        self.Wo = torch.nn.Linear(hidden_size,input_t)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self,KEY,VALUE,QUERY,return_attention=False):

        Wk = self.Wk(KEY)
        Wq = self.Wq(QUERY)
        Wv = self.Wv(VALUE)

        # scaling
        KEY_scale = torch.einsum('bij,bij->bij',KEY , Wk)
        QUERY_scale = torch.einsum('bij,bij->bij',QUERY , Wq)
        VALUE_scale = torch.einsum('bij,bij->bij',VALUE , Wv)

        # score: dot prod between KEY_scale and QUERY_scale
        d_wk = Wq.shape[-1]
        score = torch.einsum('bij,bkj->bik',KEY_scale,QUERY_scale)/math.sqrt(d_wk)

        # attention
        self.attention_weights = torch.nn.functional.softmax(score, dim=1)

        # context
        context = torch.einsum('bik,bij->bjk',self.attention_weights,VALUE_scale)

        # output
        res = self.dropout(self.Wo(context))

        if return_attention:
            return res, self.attention_weights
        else:
            return res

######
###### additive attention
######

class NW6(torch.nn.Module):
    def __init__(self, input_size,input_v, hidden_size,dropout=0.1):
        super(NW6, self).__init__()

        self.W = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, input_size))

        self.Ws = torch.nn.Sequential(
           torch.nn.Linear(input_size, hidden_size),
           torch.nn.ReLU(),
           torch.nn.Linear(hidden_size, hidden_size),
           torch.nn.ReLU(),
           torch.nn.Linear(hidden_size, input_v))

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, KEY,VALUE,QUERY,return_attention=False):

        Wk = self.W(KEY)
        Wq = self.W(QUERY)

        # score
        score = self.Ws(torch.tanh(Wk+Wq))

        self.attention_weights = torch.nn.functional.softmax(-torch.pow(score,2) / 2, dim=1)

        # context
        context = torch.einsum('bij,bij->bij',self.attention_weights,VALUE)

        # output
        res = self.dropout(context)

        if return_attention:
            return res, self.attention_weights
        else:
            return res


###############################
# MULTI-HEAD ATTENTION        #
###############################

class mha0(torch.nn.Module):
    """
    as in https://arxiv.org/abs/1706.03762
    In addition, We assume VALUE_X and VALUE_Y belonging to one identic process.
    So the scaling of KEY and QUERY is applied with one identical weight `W`.
    """

    def __init__(self,input_k, input_v, hidden_size, output_size, num_heads=1,dropout=0.1):
        super(mha0, self).__init__()

        assert hidden_size % num_heads == 0, "Hidden size must be 0 modulo number of heads."

        self.input_k = input_k
        self.input_v = input_v
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # W_keys
        self.W = torch.nn.Linear(input_k, hidden_size)

        # W_values
        self.Wv = torch.nn.Linear(input_v, hidden_size)

        # W_ouput
        self.Wo = torch.nn.Linear(hidden_size,output_size)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self,KEY,VALUE,QUERY,return_attention=False):

        Wk = self.W(KEY) # bij
        Wq = self.W(QUERY) # bkj
        Wv = self.Wv(VALUE) # bij

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
        res = self.dropout(self.Wo(output)) # bkl

        if return_attention:
            return res, self.attention_weights
        else:
            return res

class mha1(torch.nn.Module):
    """
    as in https://arxiv.org/abs/1706.03762
    """

    def __init__(self,input_k, input_q, input_v, hidden_size, output_size, num_heads=1,dropout=0.1):
        super(mha1, self).__init__()

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

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self,KEY,VALUE,QUERY,return_attention=False):

        Wk = self.Wk(KEY) # bij: 64x51x8
        Wq = self.Wq(QUERY) # bkj: 64x20x8
        Wv = self.Wv(VALUE) # bij: 64x51x8

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
        res = self.dropout(self.Wo(output)) # bkl

        if return_attention:
            return res, self.attention_weights
        else:
            return res

class mha2(torch.nn.Module):
    """
    """

    def __init__(self,input_k, input_q, input_v, hidden_size, output_size, num_heads=1,dropout=0.1):
        super(mha2, self).__init__()

        assert hidden_size % num_heads == 0, "Hidden size must be 0 modulo number of heads."

        self.input_k = input_k
        self.input_q = input_q
        self.input_v = input_v
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # W_keys
        self.Wk = torch.nn.Sequential(
            torch.nn.Linear(input_k, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size))

        # W_queries
        self.Wq = torch.nn.Sequential(
            torch.nn.Linear(input_q, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size))

        # W_values
        self.Wv = torch.nn.Sequential(
            torch.nn.Linear(input_v, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size))

        # W_ouput
        self.Wo = torch.nn.Linear(hidden_size,output_size)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self,KEY,VALUE,QUERY,return_attention=False):

        Wk = self.Wk(KEY) # bij: 64x51x8
        Wq = self.Wq(QUERY) # bkj: 64x20x8
        Wv = self.Wv(VALUE) # bij: 64x51x8

        ##############
        # multi-head #
        ##############

        # reshaping into a multi-head
        Wk = torch.reshape(Wk,(Wk.shape[0], Wk.shape[1], int(Wk.shape[2]/self.num_heads), self.num_heads)) #bi(j/h)h
        Wk = Wk.permute(0,3,1,2) # bhi(j/h)
        Wq = torch.reshape(Wq,(Wq.shape[0], Wq.shape[1], int(Wq.shape[2]/self.num_heads), self.num_heads)) #bk(j/h)h
        Wq = Wq.permute(0,3,1,2) # bhk(j/h)
        Wv = torch.reshape(Wv,(Wv.shape[0], Wv.shape[1], int(Wv.shape[2]/self.num_heads), self.num_heads)) #bi(j/h)h
        Wv = Wv.permute(0,3,1,2) # bhi(j/h)

        # score
        score = torch.cdist(Wk,Wq, p=1)

        # attention weight
        self.attention_weights = torch.nn.functional.softmax(-torch.pow(score,2) / 2, dim=1)

        # context
        output = torch.einsum('bhik,bhij->bhjk',self.attention_weights,Wv) # bh(j/h)k

        #######################
        # Concatenating heads #
        #######################
        output = output.permute(0, 2, 1, 3)  # b(j/h)hk
        output = output.reshape(output.shape[0], output.shape[1]*output.shape[2], output.shape[3]) # bjk
        output = torch.transpose(output,2,1) # bkj

        # prediction
        res = self.dropout(self.Wo(output)) # bkl

        if return_attention:
            return res, self.attention_weights
        else:
            return res
