import torch

########
######## Multi-dimension Attention with Distance as Score (MADS)
########
class mads(torch.nn.Module):

    def __init__(self,device,type,kernel,input_k):
        super(mads, self).__init__()

        if type == "krig":
            self.attention = class_krig(device,kernel)
        elif type == "nwd":
            self.attention = class_nwd(device,kernel)
        else:
             raise ValueError("Attention type not recognized")

        self.W = torch.ones(input_k, requires_grad=True, device=self.attention.device)

    def forward(self,KEY,VALUE,QUERY,return_attention=False):

        Wk = torch.einsum('bij,j->bij',torch.ones(KEY.shape,device=self.attention.device),self.W)
        Wq = torch.einsum('bij,j->bij',torch.ones(QUERY.shape,device=self.attention.device),self.W)

        # scaling
        KEY_scaled = torch.einsum('bij,bij->bij',KEY , Wk)
        QUERY_scaled = torch.einsum('bij,bij->bij',QUERY , Wq)

        res = self.attention.pred(KEY_scaled,VALUE,QUERY_scaled)

        if return_attention:
            return res, self.attention.weights
        else:
            return res

class mads2(torch.nn.Module):

    def __init__(self,device,type,kernel,input_k,input_v):
        super(mads2, self).__init__()

        if type == "krig":
            self.attention = class_krig(device,kernel)
        elif type == "nwd":
            self.attention = class_nwd(device,kernel)
        else:
             raise ValueError("Attention type not recognized")

        self.W = torch.ones(input_k, requires_grad=True, device=self.attention.device)
        self.Wo = torch.ones(input_v, requires_grad=True, device=self.attention.device)

    def forward(self,KEY,VALUE,QUERY,return_attention=False):

        Wk = torch.einsum('bij,j->bij',torch.ones(KEY.shape, device=self.attention.device),self.W)
        Wq = torch.einsum('bij,j->bij',torch.ones(QUERY.shape, device=self.attention.device),self.W)

        # scaling
        KEY_scaled = torch.einsum('bij,bij->bij',KEY , Wk)
        QUERY_scaled = torch.einsum('bij,bij->bij',QUERY , Wq)

        context = self.attention.pred(KEY_scaled,VALUE,QUERY_scaled)

        res = torch.einsum('bij,bij->bij',context , Wo)

        if return_attention:
            return res, self.attention.weights
        else:
            return res

class mads3(torch.nn.Module):

    def __init__(self,device,type,kernel,input_k,input_q,input_v):
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

    def forward(self,KEY,VALUE,QUERY,return_attention=False):

        Wk = torch.einsum('bij,j->bij',torch.ones(KEY.shape),self.Wk)
        Wq = torch.einsum('bij,j->bij',torch.ones(QUERY.shape),self.Wq)

        # scaling
        KEY_scaled = torch.einsum('bij,bij->bij',KEY , Wk)
        QUERY_scaled = torch.einsum('bij,bij->bij',QUERY , Wq)

        context = self.attention.pred(KEY_scaled,VALUE,QUERY_scaled)

        res = torch.einsum('bij,bij->bij',context , Wo)

        if return_attention:
            return res, self.attention.weights
        else:
            return res

class madsnn(torch.nn.Module):

    def __init__(self,device,type,kernel,input_k, hidden_size,dropout=0.1):
        super(madsnn, self).__init__()

        if type == "krig":
            self.attention = class_krig(device,kernel)
        elif type == "nwd":
            self.attention = class_nwd(device,kernel)
        else:
             raise ValueError("Attention type not recognized")

        self.W = torch.nn.Sequential(
            torch.nn.Linear(input_k, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, input_k))

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self,KEY,VALUE,QUERY,return_attention=False):

        Wk = self.W(KEY)
        Wq = self.W(QUERY)

        # scaling
        KEY_scaled = torch.einsum('bij,bij->bij',KEY , Wk)
        QUERY_scaled = torch.einsum('bij,bij->bij',QUERY , Wq)

        context = self.attention.pred(KEY_scaled,VALUE,QUERY_scaled)

        #output
        res = self.dropout(context)

        if return_attention:
            return res, self.attention.weights
        else:
            return res

class madsnn2(torch.nn.Module):

    def __init__(self,device,type,kernel,input_k, input_q, input_v, hidden_size,dropout=0.1):
        super(madsnn2, self).__init__()

        if type == "krig":
            self.attention = class_krig(device,kernel)
        elif type == "nwd":
            self.attention = class_nwd(device,kernel)
        else:
             raise ValueError("Attention type not recognized")

        self.W = torch.nn.Sequential(
            torch.nn.Linear(input_k, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, input_k))

        self.Wo = torch.nn.Sequential(
            torch.nn.Linear(input_q, hidden_size), # <- input_q
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, input_v))

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self,KEY,VALUE,QUERY,return_attention=False):

        Wk = self.W(KEY)
        Wq = self.W(QUERY)
        Wo = self.Wo(QUERY)  # with QUERY

        # scaling
        KEY_scaled = torch.einsum('bij,bij->bij',KEY , Wk)
        QUERY_scaled = torch.einsum('bij,bij->bij',QUERY , Wq)

        context = self.attention.pred(KEY_scaled,VALUE,QUERY_scaled)

        context_scaled = torch.einsum('bij,bij->bij',context , Wo)

        # output
        res = self.dropout(context_scaled)

        if return_attention:
            return res, self.attention.weights
        else:
            return res

class madsnn3(torch.nn.Module):

    def __init__(self,device,type,kernel,input_k,input_q, input_v, hidden_size,dropout=0.1):
        super(madsnn3, self).__init__()

        if type == "krig":
            self.attention = class_krig(device,kernel)
        elif type == "nwd":
            self.attention = class_nwd(device,kernel)
        else:
             raise ValueError("Attention type not recognized")

        self.Wk = torch.nn.Sequential(
            torch.nn.Linear(input_k, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, input_k))

        self.Wq = torch.nn.Sequential(
            torch.nn.Linear(input_q, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, input_q))

        self.Wo = torch.nn.Sequential(
            torch.nn.Linear(input_q, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, input_v))

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self,KEY,VALUE,QUERY,return_attention=False):

        Wk = self.Wk(KEY)
        Wq = self.Wq(QUERY)
        Wo = self.Wo(QUERY)  # with QUERY

        # scaling
        KEY_scaled = torch.einsum('bij,bij->bij',KEY , Wk)
        QUERY_scaled = torch.einsum('bij,bij->bij',QUERY , Wq)

        context = self.attention.pred(KEY_scaled,VALUE,QUERY_scaled)

        context_scaled = torch.einsum('bij,bij->bij',context , Wo)

        # output
        res = self.dropout(context_scaled)

        if return_attention:
            return res, self.attention.weights
        else:
            return res

class class_krig():
    def __init__(self,device,kernel="exp"):
        self.device = device
        self.kernel = kernel

    def variog(self,dist):
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
        gamma_ij
        KEY: coordinates (x,y,...) of dim (nbatch,nbpoints,n coords)
        '''

        # Euclidian scaled distance matrix between points [x,y]_i, i:1->n and points [x,y]_star
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
        gamma_jstar
        KEY: coordinates (x,y,...) of dim (nbatch,nbpoints,n coords)
        QUERY: coordinates (x,y,...) of dim (nbatch,nbpoints,n coords)
        '''

        # Euclidian scaled distance matrix between points [x,y]_i, i:1->n and points [x,y]_star
        dist = torch.cdist(KEY,QUERY, p=2)

        # exponential variogram of variance equal to 1
        res = self.variog(dist)
        # Lagrangian multiplier

        ## tensor [b,1,N]
        lm = torch.tensor((), dtype=torch.float64).to(self.device)
        lm = lm.new_zeros((res.shape[0],1,res.shape[2]))
        res = torch.cat((res,lm),1)

        #res = torch.reshape(res,(res.shape[0],res.shape[2],res.shape[1]))

        return(res)

    def attention(self,KEY,QUERY):
        '''
        KEY: coordinates (x,y,...) of dim (nbatch,nbpoints,n coords)
        QUERY: coordinates (x,y,...) of dim (nbatch,nbpoints,n coords)
        Solving this optimization problem g_ij^-1 . g_jstar (w/ Lagrange multipliers) results in the kriging system
        '''

        g_ij = self.gamma_ij(KEY)

        g_jstar = self.gamma_jstar(KEY,QUERY)

        # https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html#torch.linalg.lstsq
        #rem: torch.linalg.solve gives double
        res = torch.linalg.lstsq(g_ij,g_jstar).solution.float()

        return(res)

    def pred(self,KEY,VALUE,QUERY):
        self.weights = self.attention(KEY,QUERY)[:,range(KEY.shape[1])]
        res = torch.einsum('bij,bik->bjk',self.weights,VALUE)
        return(res)

class class_nwd():
    def __init__(self,device,kernel="gauss"):
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
        KEY: coordinates (x,y,...) of dim (nbatch,nbpoints,n coords)
        QUERY: coordinates (x,y,...) of dim (nbatch,nbpoints,n coords)
        '''
        # dist
        dist = torch.cdist(KEY,QUERY, p=2) # here, p=2

        res = torch.nn.functional.softmax(self.kern(dist), dim=1)

        return(res)

    def pred(self,KEY,VALUE,QUERY):

        # attention
        self.weights = self.attention(KEY,QUERY)[:,range(KEY.shape[1])]

        # context
        res = torch.einsum('bij,bik->bjk',self.weights,VALUE)

        return(res)

########
######## Scaled-dot-prod attention
########
class NWnnSDP(torch.nn.Module):
    """
    Score is based on scaled dot product of scaled coordinates.
    Weights scale the coordinates and are similar to non-uniform ranges for the variogram.
    We assume VALUE_X and VALUE_Y belonging to one identic process.
    So the scaling of KEY and QUERY is applied with one identical weight `W`.
    Attention follows NW kernel with Gaussian Kernel.
    """
    def __init__(self, input_size,input_v, input_t, hidden_size,dropout=0.1):
        super(NWnnSDP, self).__init__()

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

        # dist: dot prod between Wk and Wq
        d_wk = KEY_scale.shape[-1]
        dist = torch.einsum('bij,bkj->bik',KEY_scale,QUERY_scale)/math.sqrt(d_wk)

        # attention
        self.weights = torch.nn.functional.softmax(-torch.pow(dist,2) / 2, dim=1)

        # context
        context = torch.einsum('bik,bij->bkj',self.weights,VALUE_scale)

        # output
        res = self.dropout(self.Wo(context))

        if return_attention:
            return res, self.weights
        else:
            return res


class NWnnSDP2(torch.nn.Module):
    def __init__(self,input_k, input_q, input_v, input_t, hidden_size, dropout=0.1):
        super(NWnnSDP2, self).__init__()

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

        # dist: dot prod between KEY_scale and QUERY_scale
        d_wk = Wq.shape[-1]
        dist = torch.einsum('bij,bkj->bik',KEY_scale,QUERY_scale)/math.sqrt(d_wk)

        # attention
        self.weights = torch.nn.functional.softmax(dist, dim=1)

        # context
        context = torch.einsum('bik,bij->bjk',self.weights,VALUE_scale)

        # output
        res = self.dropout(self.Wo(context))

        if return_attention:
            return res, self.weights
        else:
            return res

######
###### additive attention
######
class NWnnAdd(torch.nn.Module):
    def __init__(self, input_size,input_v, hidden_size,dropout=0.1):
        super(NWnnAdd, self).__init__()

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

        # dist
        dist = self.Ws(torch.tanh(Wk+Wq))

        self.weights = torch.nn.functional.softmax(-torch.pow(dist,2) / 2, dim=1)

        # context
        context = torch.einsum('bij,bij->bij',self.weights,VALUE)

        # output
        res = self.dropout(context)

        if return_attention:
            return res, self.weights
        else:
            return res
