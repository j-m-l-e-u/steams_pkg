import torch

class krignn(torch.nn.Module):

    def __init__(self,device,vmodel,input_k, hidden_size,dropout=0.1):
        super(krignn, self).__init__()

        self.krig = class_krig(device,vmodel)

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
        KEY_scale = torch.einsum('bij,bij->bij',KEY , Wk)
        QUERY_scale = torch.einsum('bij,bij->bij',QUERY , Wq)

        context = self.krig.krig_pred(KEY_scale,VALUE,QUERY_scale)

        #output
        res = self.dropout(context)

        if return_attention:
            return res, self.krig.attention_weights
        else:
            return res


class krignn2(torch.nn.Module):

    def __init__(self,device,vmodel,input_k, input_q, input_v, hidden_size,dropout=0.1):
        super(krignn2, self).__init__()

        self.krig = class_krig(device,vmodel)

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
        KEY_scale = torch.einsum('bij,bij->bij',KEY , Wk)
        QUERY_scale = torch.einsum('bij,bij->bij',QUERY , Wq)

        context = self.krig.krig_pred(KEY_scale,VALUE,QUERY_scale)

        context_scale = torch.einsum('bij,bij->bij',context , Wo)

        # output
        res = self.dropout(context_scale)

        if return_attention:
            return res, self.krig.attention_weights
        else:
            return res

class krignn3(torch.nn.Module):

    def __init__(self,device,vmodel,input_k,input_q, input_v, hidden_size,dropout=0.1):
        super(krignn3, self).__init__()

        self.krig = class_krig(device,vmodel)

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
        KEY_scale = torch.einsum('bij,bij->bij',KEY , Wk)
        QUERY_scale = torch.einsum('bij,bij->bij',QUERY , Wq)

        context = self.krig.krig_pred(KEY_scale,VALUE,QUERY_scale)

        context_scale = torch.einsum('bij,bij->bij',context , Wo)

        # output
        res = self.dropout(context_scale)

        if return_attention:
            return res, self.krig.attention_weights
        else:
            return res

class class_krig():
    def __init__(self,device,vmodel="exp"):
        self.device = device
        self.vmodel = vmodel

    def variog(self,dist):
        if self.vmodel == "exp":
            res = torch.tensor((), dtype=torch.float64).to(self.device)
            res = res.new_ones((dist.shape[0],dist.shape[1],dist.shape[2])) - torch.exp(-dist)
        elif self.vmodel == "gauss":
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

    def k_weight(self,KEY,QUERY):
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

    def krig_pred(self,KEY,VALUE,QUERY):
        self.attention_weights = self.k_weight(KEY,QUERY)[:,range(KEY.shape[1])]
        res = torch.einsum('bij,bik->bjk',self.attention_weights,VALUE)
        return(res)
