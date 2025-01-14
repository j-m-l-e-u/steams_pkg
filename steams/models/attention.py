import torch
from torch import nn
import math

########
######## Multi-dimension Attention with Distance as Score (MADS)
########
class mads(nn.Module):
    def __init__(self,device,type,kernel,input_k):
        '''
        mads is an adaptive distance attention model. It is either based on the kriging equation system or the Nadaray-Watson kernel.
        It assumes both network X and Y observing the same phenomenon.
        No deep learning is involved in this model

        Args:
            device:
            Determined with torch.device()

            type:
            Determines either the krigin sytem or the Nadaraya-Watson Kernel: 'krig' or 'nwd'.

            kernel:
            Deternines either Gaussian kernel or Exponential; 'gauss', 'exp'.

            input_k:
            Number of keys as input.
        '''
        super(mads, self).__init__()

        if type == "krig":
            self.attention = class_krig(device,kernel)
        elif type == "nwd":
            self.attention = class_nwd(device,kernel)
        else:
             raise ValueError("Attention type not recognized")

        self.W = torch.ones(input_k, requires_grad=True, device=self.attention.device)

    def forward(self,KEY,VALUE,QUERY):

        Wk = torch.einsum('bij,j->bij',torch.ones(KEY.shape,device=self.attention.device),self.W)
        Wq = torch.einsum('bij,j->bij',torch.ones(QUERY.shape,device=self.attention.device),self.W)

        # scaling
        KEY_scaled = torch.einsum('bij,bij->bij',KEY , Wk)
        QUERY_scaled = torch.einsum('bij,bij->bij',QUERY , Wq)
        res = self.attention.pred(KEY_scaled,VALUE,QUERY_scaled)

        return res

class mads2(nn.Module):
    def __init__(self,device,type,kernel,input_k,input_v):
        '''
        mads2 is an adaptive distance attention model. It is either based on the kriging equation system or the Nadaray-Watson kernel.
        It assumes both network X and Y observing the same phenomenon. In addition, a multiplicative parameter is used to predict the output.
        No deep learning is involved in this model

        Args:
            device:
            Determined with torch.device()

            type:
            Determines either the krigin sytem or the Nadaraya-Watson Kernel: 'krig' or 'nwd'.

            kernel:
            Deternines either Gaussian kernel or Exponential; 'gauss', 'exp'.

            input_k:
            Number of keys as input.

            input_v
            Number of values as input.
        '''
        super(mads2, self).__init__()

        if type == "krig":
            self.attention = class_krig(device,kernel)
        elif type == "nwd":
            self.attention = class_nwd(device,kernel)
        else:
             raise ValueError("Attention type not recognized")

        self.W = torch.ones(input_k, requires_grad=True, device=self.attention.device)
        self.Wo = torch.ones(input_v, requires_grad=True, device=self.attention.device)

    def forward(self,KEY,VALUE,QUERY):

        Wk = torch.einsum('bij,j->bij',torch.ones(KEY.shape, device=self.attention.device),self.W)
        Wq = torch.einsum('bij,j->bij',torch.ones(QUERY.shape, device=self.attention.device),self.W)
        Wo = torch.einsum('bij,j->bij',torch.ones(VALUE.shape, device=self.attention.device),self.Wo)

        # scaling
        KEY_scaled = torch.einsum('bij,bij->bij',KEY , Wk)
        QUERY_scaled = torch.einsum('bij,bij->bij',QUERY , Wq)
        context = self.attention.pred(KEY_scaled,VALUE,QUERY_scaled)

        res = torch.einsum('bij,bij->bij',context , Wo)

        return res

class mads3(nn.Module):
    def __init__(self,device,type,kernel,input_k,input_q,input_v):
        '''
        mads3 is an adaptive distance attention model. It is either based on the kriging equation system or the Nadaray-Watson kernel.
        Network X and network Y observes two different phenomena. In addition, a multiplicative parameter is used to predict the output.
        No deep learning is involved in this model

        Args:
            device:
            Determined with torch.device()

            type:
            Determines either the krigin sytem or the Nadaraya-Watson Kernel: 'krig' or 'nwd'.

            kernel:
            Deternines either Gaussian kernel or Exponential; 'gauss', 'exp'.

            input_k:
            Number of keys as input.

            input_q:
            Number of queries as input.

            input_v
            Number of values as input.
        '''
        super(mads3, self).__init__()

        if type == "krig":
            self.attention = class_krig(device,kernel)
        elif type == "nwd":
            self.attention = class_nwd(device,kernel)
        else:
             raise ValueError("Attention type not recognized")

        self.Wk = torch.ones(input_k, requires_grad=True)
        self.Wq = torch.ones(input_q, requires_grad=True)
        self.Wo = torch.ones(input_v, requires_grad=True)

    def forward(self,KEY,VALUE,QUERY):

        Wk = torch.einsum('bij,j->bij',torch.ones(KEY.shape, device=self.attention.device),self.Wk)
        Wq = torch.einsum('bij,j->bij',torch.ones(QUERY.shape, device=self.attention.device),self.Wq)
        Wo = torch.einsum('bij,j->bij',torch.ones(VALUE.shape, device=self.attention.device),self.Wo)

        # scaling
        KEY_scaled = torch.einsum('bij,bij->bij',KEY , Wk)
        QUERY_scaled = torch.einsum('bij,bij->bij',QUERY , Wq)
        context = self.attention.pred(KEY_scaled,VALUE,QUERY_scaled)

        res = torch.einsum('bij,bij->bij',context , Wo)

        return res

class madsnn(nn.Module):
    def __init__(self,device,type,kernel,input_k, hidden_size,dropout=0.1):
        '''
        madsnn is an adaptive distance attention model. It is either based on the kriging equation system or the Nadaray-Watson kernel.
        It assumes both network X and Y observing the same phenomenon.
        Multilayer perceptrons are involved into the learnable parameters.

        Args:
            device:
            Determined with torch.device()

            type:
            Determines either the krigin sytem or the Nadaraya-Watson Kernel: 'krig' or 'nwd'.

            kernel:
            Deternines either Gaussian kernel or Exponential; 'gauss', 'exp'.

            input_k:
            Number of keys as input.

            hidden_size:
            Number of hidden layers.

            dropout:
            Probability for the dropout; By default, dropout = 0.1.
        '''
        super(madsnn, self).__init__()

        if type == "krig":
            self.attention = class_krig(device,kernel)
        elif type == "nwd":
            self.attention = class_nwd(device,kernel)
        else:
             raise ValueError("Attention type not recognized")

        self.W = nn.Sequential(
            nn.Linear(input_k, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_k))

        self.dropout = nn.Dropout(dropout)

    def forward(self,KEY,VALUE,QUERY):

        Wk = self.W(KEY)
        Wq = self.W(QUERY)

        # scaling
        # KEY_scaled = torch.einsum('bij,bij->bij',KEY , Wk)
        # QUERY_scaled = torch.einsum('bij,bij->bij',QUERY , Wq)
        # context = self.attention.pred(KEY_scaled,VALUE,QUERY_scaled)

        context = self.attention.pred(Wk,VALUE,Wq)

        #output
        res = self.dropout(context)

        return res

class madsnn2(nn.Module):
    def __init__(self,device,type,kernel,input_k, input_q, input_v, hidden_size,dropout=0.1):
        '''
        madsnn2 is an adaptive distance attention model. It is either based on the kriging equation system or the Nadaray-Watson kernel.
        It assumes both network X and Y observing the same phenomenon. In addition, a multiplicative parameter is used to predict the output.
        Multilayer perceptrons are involved into the learnable parameters.

        Args:
            device:
            Determined with torch.device()

            type:
            Determines either the krigin sytem or the Nadaraya-Watson Kernel: 'krig' or 'nwd'.

            kernel:
            Deternines either Gaussian kernel or Exponential; 'gauss', 'exp'.

            input_k:
            Number of keys as input.

            input_q:
            Number of queries as input.

            input_v:
            Number of values as input.

            hidden_size:
            Number of hidden layers.

            dropout:
            Probability for the dropout; By default, dropout = 0.1.
        '''
        super(madsnn2, self).__init__()

        if type == "krig":
            self.attention = class_krig(device,kernel)
        elif type == "nwd":
            self.attention = class_nwd(device,kernel)
        else:
             raise ValueError("Attention type not recognized")

        self.W = nn.Sequential(
            nn.Linear(input_k, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_k))

        self.Wo = nn.Sequential(
            nn.Linear(input_q, hidden_size), # <- input_q
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_v))

        self.dropout = nn.Dropout(dropout)

    def forward(self,KEY,VALUE,QUERY):

        Wk = self.W(KEY)
        Wq = self.W(QUERY)
        Wo = self.Wo(QUERY)  # with QUERY

        # # scaling
        # KEY_scaled = torch.einsum('bij,bij->bij',KEY , Wk)
        # QUERY_scaled = torch.einsum('bij,bij->bij',QUERY , Wq)
        # context = self.attention.pred(KEY_scaled,VALUE,QUERY_scaled)

        context = self.attention.pred(Wk,VALUE,Wq)

        context_scaled = torch.einsum('bij,bij->bij',context , Wo)

        # output
        res = self.dropout(context_scaled)

        return res

class madsnn3(nn.Module):
    def __init__(self,device,type,kernel,input_k,input_q, input_v, hidden_size,dropout=0.1):
        '''
        madsnn3 is an adaptive distance attention model. It is either based on the kriging equation system or the Nadaray-Watson kernel.
        Network X and network Y observes two different phenomena. In addition, a multiplicative parameter is used to predict the output.
        Multilayer perceptrons are involved into the learnable parameters.

        Args:
            device:
            Determined with torch.device()

            type:
            Determines either the krigin sytem or the Nadaraya-Watson Kernel: 'krig' or 'nwd'.

            kernel:
            Deternines either Gaussian kernel or Exponential; 'gauss', 'exp'.

            input_k:
            Number of keys as input.

            input_q:
            Number of queries as input.

            input_v:
            Number of values as input.

            hidden_size:
            Number of hidden layers.

            dropout:
            Probability for the dropout; By default, dropout = 0.1.
        '''
        super(madsnn3, self).__init__()

        if type == "krig":
            self.attention = class_krig(device,kernel)
        elif type == "nwd":
            self.attention = class_nwd(device,kernel)
        else:
             raise ValueError("Attention type not recognized")

        self.Wk = nn.Sequential(
            nn.Linear(input_k, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_k))

        self.Wq = nn.Sequential(
            nn.Linear(input_q, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_q))

        self.Wo = nn.Sequential(
            nn.Linear(input_q, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_v))

        self.dropout = nn.Dropout(dropout)

    def forward(self,KEY,VALUE,QUERY):

        Wk = self.Wk(KEY)
        Wq = self.Wq(QUERY)
        Wo = self.Wo(QUERY)  # with QUERY

        # scaling
        context = self.attention.pred(Wk,VALUE,Wq)

        context_scaled = torch.einsum('bij,bij->bij',context , Wo)

        # output
        res = self.dropout(context_scaled)

        return res

class madsnn4(nn.Module):
    def __init__(self,device,type,kernel,input_k,input_q, input_v, hidden_size,dropout=0.1):
        '''
        madsnn3 is an adaptive distance attention model. It is either based on the kriging equation system or the Nadaray-Watson kernel.
        Network X and network Y observes two different phenomena. In addition, a multiplicative parameter is used to predict the output.
        Multilayer perceptrons are involved into the learnable parameters.

        Args:
            device:
            Determined with torch.device()

            type:
            Determines either the krigin sytem or the Nadaraya-Watson Kernel: 'krig' or 'nwd'.

            kernel:
            Deternines either Gaussian kernel or Exponential; 'gauss', 'exp'.

            input_k:
            Number of keys as input.

            input_q:
            Number of queries as input.

            input_v:
            Number of values as input.

            hidden_size:
            Number of hidden layers.

            dropout:
            Probability for the dropout; By default, dropout = 0.1.
        '''
        super(madsnn4, self).__init__()

        if type == "krig":
            self.attention = class_krig(device,kernel)
        elif type == "nwd":
            self.attention = class_nwd(device,kernel)
        else:
             raise ValueError("Attention type not recognized")

        self.Wk = nn.Sequential(
            nn.Linear(input_k, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size))

        self.Wq = nn.Sequential(
            nn.Linear(input_q, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size))

        self.Wv = nn.Sequential(
            nn.Linear(input_v, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size))
        
        self.Wo = nn.Linear(hidden_size,input_v)

        self.dropout = nn.Dropout(dropout)

    def forward(self,KEY,VALUE,QUERY):

        Wk = self.dropout(self.Wk(KEY))
        Wq = self.dropout(self.Wq(QUERY))
        Wv = self.dropout(self.Wv(VALUE))

        # context
        context = self.attention.pred(Wk,Wv,Wq)

        # output
        res = self.dropout(self.Wo(context)) # bkl

        return res

################
## kriging    ##
################
class class_krig():
    def __init__(self,device,kernel="exp"):
        '''
        class_krig is an adaptive distance attention model based on the kriging equation system.
        Is solves the Ordinary Kriging equation system. It thus assume the mean and the variance constant and unknown.

        Args:
            device:
            Determined with torch.device()

            kernel:
            Deternines either Gaussian kernel or Exponential; 'gauss', 'exp'.
        '''
        self.device = device
        self.kernel = kernel

    def variog(self,dist):
        '''
        Processes the variogram with a scaled distance tensor. The variogram model is determined by the kernel parameter.
        Args:
            dist: 3-dimension tensor.
        '''
        if self.kernel == "exp":
            res = torch.tensor((), dtype=torch.float64).to(self.device)
            res = res.new_ones((dist.shape[0],dist.shape[1],dist.shape[2])) - torch.exp(-dist)
        elif self.kernel == "gauss":
            res = torch.tensor((), dtype=torch.float64).to(self.device)
            res = res.new_ones((dist.shape[0],dist.shape[1],dist.shape[2])) - torch.exp(-torch.pow(dist,2))
        return res

    def get_dist_ij(self,KEY):
        # Euclidian scaled distance matrix between points [x,y]_i, i:1->n and points [x,y]_star
        dist = torch.cdist(KEY,KEY, p=2)
        res = res.to(self.device)
        return(res)

    def gamma_ij(self,KEY):
        '''
        Processes the matrix variogram in-between each pair of keys.

        Args:
            KEY: tensor key of dimension (nbatch,nbpoints,input_k)
        '''

        # Euclidian scaled distance matrix between points [x,y]_i, i:1->n and points [x,y]_j, j:1->n
        dist = torch.cdist(KEY,KEY, p=2)

        # variogram of variance equal to 1
        res = self.variog(dist)

        # Lagrangian multiplier
        ## tensor [b,i,N]
        lm1 = torch.tensor((), dtype=torch.float64).to(self.device)
        lm1 = lm1.new_ones((dist.shape[0],dist.shape[1],1))
        ## tensor [b,N,j]
        lm2 = torch.tensor((), dtype=torch.float64).to(self.device)
        lm2 = lm2.new_ones((dist.shape[0],1,dist.shape[1]))
        ## tensor [b,N,N]
        lm3 = torch.tensor((), dtype=torch.float64).to(self.device)
        lm3 = lm3.new_zeros((dist.shape[0],1,1))

        res = torch.cat((res,lm1),2)

        lm4 = torch.cat((lm2,lm3),2)
        res = torch.cat((res,lm4),1)

        return(res)

    def gamma_jstar(self,KEY,QUERY):
        '''
        Processes the matrix variogram in-between each pair of keys and queries.

        Args:
            KEY: tensor key of dimension (nbatch,nbpoints,input_k)

            QUERY: tensor query of dimension (nbatch,nbpoints,input_q)
        '''

        # Euclidian scaled distance matrix between points [x,y]_i, i:1->n and points [x,y]_k, k:1->m
        dist = torch.cdist(KEY,QUERY, p=2)

        # variogram of variance equal to 1
        res = self.variog(dist)

        # Lagrangian multiplier
        ## tensor [b,1,N]
        lm = torch.tensor((), dtype=torch.float64).to(self.device)
        lm = lm.new_zeros((res.shape[0],1,res.shape[2]))
        res = torch.cat((res,lm),1)

        return(res)

    def attention(self,KEY,QUERY):
        '''
        Solving the Ordinary Kriging equation system with the least square method.

        Args:
            KEY: tensor key of dimension (nbatch,nbpoints,input_k)

            QUERY: tensor query of dimension (nbatch,nbpoints,input_q)
        '''

        g_ij = self.gamma_ij(KEY)

        g_jstar = self.gamma_jstar(KEY,QUERY)

        # https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html#torch.linalg.lstsq
        #rem: torch.linalg.solve gives double
        res = torch.linalg.lstsq(g_ij,g_jstar).solution.float()

        return(res)

    def pred(self,KEY,VALUE,QUERY):
        '''
        Prediction using the weight of Ordinary Kriging equation.

        Args:
            KEY: tensor key of dimension (nbatch,nbpoints,input_k)

            VALUE: tensor value of dimension (nbatch,nbpoints,input_v)

            QUERY: tensor query of dimension (nbatch,nbpoints,input_q)
        '''
        self.weights = self.attention(KEY,QUERY)[:,range(KEY.shape[1])]
        res = torch.einsum('bij,bik->bjk',self.weights,VALUE)
        return(res)

############################################
## Nadaraya-Watson with distance as score ##
############################################
class class_nwd():
    def __init__(self,device,kernel="gauss"):
        '''
        class_nwd is an adaptive distance attention model based on the Nadaray-Watson kernel.

        Args:
            device:
            Determined with torch.device()

            kernel:
            Deternines either Gaussian kernel or Exponential; 'gauss', 'exp'.
        '''
        self.device = device
        self.kernel = kernel

    def kern(self,dist):
        if self.kernel == "exp":
            res = -dist/2
        elif self.kernel == "gauss":
            res = -torch.pow(dist,2)/2
        return res

    def attention(self,KEY,QUERY):
        '''
        Provides the attention using the softmax function.

        Args:
            KEY: tensor key of dimension (nbatch,nbpoints,input_k)

            QUERY: tensor query of dimension (nbatch,nbpoints,input_q)
        '''

        # dist
        dist = torch.cdist(KEY,QUERY, p=2) # here, p=2

        res = nn.functional.softmax(self.kern(dist), dim=1)

        return(res)

    def pred(self,KEY,VALUE,QUERY):
        '''
        Prediction.

        Args:
            KEY: tensor key of dimension (nbatch,nbpoints,input_k)

            VALUE: tensor value of dimension (nbatch,nbpoints,input_v)

            QUERY: tensor query of dimension (nbatch,nbpoints,input_q)
        '''

        # attention
        self.weights = self.attention(KEY,QUERY)[:,range(KEY.shape[1])]

        # context
        res = torch.einsum('bij,bik->bjk',self.weights,VALUE)

        return(res)

########
######## Scaled-dot-prod attention
########

class dpnn3(nn.Module):
    def __init__(self,input_k, input_v, hidden_size, dropout=0.1):
        '''
        dpnn2 is an adaptive attention model using on the dot product between the key and query tensors.
        Network X and network Y observes two different phenomena. In addition, a multiplicative parameter is used to predict the output.
        Multilayer perceptrons are involved into the learnable parameters.

        Args:
            device:
        
            input_k:
            Number of keys as input.

            input_v:
            Number of values as input.

            hidden_size:
            Number of hidden layers.

            dropout:
            Probability for the dropout; By default, dropout = 0.1.
        '''
        super(dpnn3, self).__init__()

        # W as an MLP
        self.W = nn.Sequential(
            nn.Linear(input_k, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_k),
            )

        # W_values as an MLP
        self.Wv = nn.Sequential(
            nn.Linear(input_v, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_v),
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self,KEY,VALUE,QUERY):

        # scaling
        Wk = self.dropout(self.W(KEY))
        Wq = self.dropout(self.W(QUERY))
        Wv = self.dropout(self.Wv(VALUE))

        # dist: dot prod between Wk and Wq
        d_wk = Wq.shape[-1]
        dist = torch.einsum('bij,bkj->bik',Wk,Wq)/math.sqrt(d_wk)

        # attention
        self.weights = nn.functional.softmax(dist, dim=2)

        # context
        context = torch.einsum('bik,bij->bkj',self.weights,Wv)
        
        # output
        res = self.dropout(context)

         
        return res

class dpnn4(nn.Module):
    def __init__(self,input_k, input_q, input_v, hidden_size, dropout=0.1):
        '''
        dpnn4 is an adaptive attention model using on the dot product between the key and query tensors.
        Network X and network Y observes two different phenomena. In addition, a multiplicative parameter is used to predict the output.
        Multilayer perceptrons are involved into the learnable parameters.

        Args:
            
            input_k:
            Number of keys as input.

            input_q:
            Number of queries as input.

            input_v:
            Number of values as input.

            hidden_size:
            Number of hidden layers.

            dropout:
            Probability for the dropout; By default, dropout = 0.1.
        '''
        super(dpnn4, self).__init__()

        # W_keys as an MLP
        self.Wk = nn.Sequential(
            nn.Linear(input_k, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_k),
            )

        # W_queries as an MLP
        self.Wq = nn.Sequential(
            nn.Linear(input_q, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_q),
            )

        # W_values as an MLP
        self.Wv = nn.Sequential(
            nn.Linear(input_v, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_v),
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self,KEY,VALUE,QUERY):

        # scaling
        Wk = self.dropout(self.Wk(KEY))
        Wq = self.dropout(self.Wq(QUERY))
        Wv = self.dropout(self.Wv(VALUE))

        
        # dist: dot prod between KEY_scale and QUERY_scale
        d_wk = Wq.shape[-1]
        dist = torch.einsum('bij,bkj->bik',Wk,Wq)/math.sqrt(d_wk)

        # attention
        self.weights = nn.functional.softmax(dist, dim=2)

        # context
        context = torch.einsum('bik,bij->bkj',self.weights,Wv)
        
        # output
        res = self.dropout(context)

        return res

class dpnn5(nn.Module):
    def __init__(self,input_k, input_q, input_v, hidden_size, dropout=0.1):
        '''
        dpnn5 is an adaptive attention model using on the dot product between the key and query tensors.
        Network X and network Y observes two different phenomena. In addition, a multiplicative parameter is used to predict the output.
        Multilayer perceptrons are involved into the learnable parameters.

        Args:
            
            input_k:
            Number of keys as input.

            input_q:
            Number of queries as input.

            input_v:
            Number of values as input.

            hidden_size:
            Number of hidden layers.

            dropout:
            Probability for the dropout; By default, dropout = 0.1.
        '''
        super(dpnn5, self).__init__()

        # W_keys as an MLP
        self.Wk = nn.Sequential(
            nn.Linear(input_k, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            )

        # W_queries as an MLP
        self.Wq = nn.Sequential(
            nn.Linear(input_q, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            )

        # W_values as an MLP
        self.Wv = nn.Sequential(
            nn.Linear(input_v, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            )

        # #W_ouput
        self.Wo = nn.Linear(hidden_size,input_v)

      
        self.dropout = nn.Dropout(dropout)

    def forward(self,KEY,VALUE,QUERY):

        Wk = self.dropout(self.Wk(KEY))
        Wq = self.dropout(self.Wq(QUERY))
        Wv = self.dropout(self.Wv(VALUE))

        # dist: dot prod between Wk and Wq
        d_wk = Wq.shape[-1]
        dist = torch.einsum('bij,bkj->bik',Wk,Wq)/math.sqrt(d_wk)

        # attention
        self.weights = nn.functional.softmax(dist, dim=2)

        # context
        context = torch.einsum('bik,bij->bkj',self.weights,Wv)
        
        # output
        res = self.dropout(self.Wo(context))
        
        return res

class mha(torch.nn.Module):
    
    def __init__(self,input_k, input_q, input_v, hidden_size, num_heads=1,dropout=0.1):
        '''
        mha is an adaptive multi-head attention model using on the dot product between key and query tensors.
        Network X and network Y observes two different phenomena. In addition, a multiplicative parameter is used to predict the output.
        Multilayer perceptrons are involved into the learnable parameters.
        

        Args:
            
            input_k:
            Number of keys as input.

            input_q:
            Number of queries as input.

            input_v:
            Number of values as input.

            hidden_size:
            Number of hidden layers.

            num_heads:
            Number of heads.

            dropout:
            Probability for the dropout; By default, dropout = 0.1.
        '''
        super(mha, self).__init__()

        assert hidden_size % num_heads == 0, "Hidden size must be 0 modulo number of heads."

        self.num_heads = num_heads

        # W_keys as an MLP
        self.Wk = nn.Sequential(
            nn.Linear(input_k, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            )

        # W_queries as an MLP
        self.Wq = nn.Sequential(
            nn.Linear(input_q, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            )

        # W_values as an MLP
        self.Wv = nn.Sequential(
            nn.Linear(input_v, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            )

        # W_ouput
        self.Wo = nn.Linear(hidden_size,input_v)

        self.dropout = nn.Dropout(dropout)

    def forward(self,KEY,VALUE,QUERY):

        Wk = self.dropout(self.Wk(KEY))
        Wq = self.dropout(self.Wq(QUERY))
        Wv = self.dropout(self.Wv(VALUE))


        # reshaping into a multi-head
        Wk = torch.reshape(Wk,(Wk.shape[0], Wk.shape[1], int(Wk.shape[2]/self.num_heads), self.num_heads)) #bi(j/h)h 
        Wk = Wk.permute(0,3,1,2) # bhi(j/h)
        Wq = torch.reshape(Wq,(Wq.shape[0], Wq.shape[1], int(Wq.shape[2]/self.num_heads), self.num_heads)) #bk(j/h)h 
        Wq = Wq.permute(0,3,1,2) # bhk(j/h)
        Wv = torch.reshape(Wv,(Wv.shape[0], Wv.shape[1], int(Wv.shape[2]/self.num_heads), self.num_heads)) #bi(j/h)h 
        Wv = Wv.permute(0,3,1,2) # bhi(j/h)

        # score: dot prod between Wk and Wq
        d_wk = Wq.shape[-1]
        score = torch.einsum('bhij,bhkj->bhik',Wk,Wq)/math.sqrt(d_wk) #bhik

        # attention weight
        self.weights = torch.nn.functional.softmax(score, dim=3) #bhik

        # context
        context = torch.einsum('bhik,bhij->bhkj',self.weights,Wv) # bh(j/h)k     

        # Concatenating heads 
        context = context.permute(0, 2, 1, 3)  # b(j/h)hk
        context = context.reshape(context.shape[0], context.shape[1], context.shape[2]*context.shape[3]) # bjk

        # output
        res = self.dropout(self.Wo(context)) # bkl

        return res

class dpnn6(nn.Module):
    def __init__(self,input_k, input_q, input_v, hidden_size, dropout=0.1):
        '''
        dpnn6 is an adaptive attention model using on the dot product between the key and query tensors.
        Network X and network Y observes two different phenomena. In addition, a multiplicative parameter is used to predict the output.
        Linear functions are involved into the learnable parameters.

        Args:
            
            input_k:
            Number of keys as input.

            input_q:
            Number of queries as input.

            input_v:
            Number of values as input.

            hidden_size:
            Number of hidden layers.

            dropout:
            Probability for the dropout; By default, dropout = 0.1.
        '''
        super(dpnn6, self).__init__()

        # W_keys as a linear function
        self.Wk = nn.Linear(input_k, hidden_size)

        # W_queries as a linear function
        self.Wq = nn.Linear(input_q, hidden_size)

        # W_values as a linear function
        self.Wv = nn.Linear(input_v, hidden_size)

        # final linear function
        self.linear = nn.Linear(hidden_size,input_v)

        self.dropout = nn.Dropout(dropout)

    def forward(self,KEY,VALUE,QUERY):

        Wk = self.dropout(self.Wk(KEY))
        Wq = self.dropout(self.Wq(QUERY))
        Wv = self.dropout(self.Wv(VALUE))

        # dist: dot prod between Wk and Wq
        d_wk = Wq.shape[-1]
        dist = torch.einsum('bij,bkj->bik',Wk,Wq)/math.sqrt(d_wk)

        # attention
        self.weights = nn.functional.softmax(dist, dim=2)

        # context
        context = torch.einsum('bik,bij->bkj',self.weights,Wv)
        
        # output
        res = self.dropout(self.linear(context))

        return res
