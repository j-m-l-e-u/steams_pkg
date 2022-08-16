import torch

class class_krig():
    def __init__(self,device):
        self.device = device

    def get_dist_ij(self,KEY,Wk):
        # scaling coordinates
        KEY_scale = torch.einsum('bij,bij->bij',KEY , Wk)
        # Euclidian scaled distance matrix between points [x,y]_i, i:1->n and points [x,y]_star
        dist = torch.cdist(KEY_scale,KEY_scale, p=2)
        res = res.to(self.device)
        return(res)

    def gamma_ij(self,KEY,Wk):
        '''
        gamma_ij
        KEY: coordinates (x,y,...) of dim (nbatch,nbpoints,n coords)
        Wk: range matrix of dim (nbatch,nbpoints,n coords)
        '''

        # scaling coordinates
        KEY_scale = torch.einsum('bij,bij->bij',KEY , Wk)

        # Euclidian scaled distance matrix between points [x,y]_i, i:1->n and points [x,y]_star
        dist = torch.cdist(KEY_scale,KEY_scale, p=2)

        # exponential variogram of variance equal to 1
        res = torch.tensor((), dtype=torch.float64).to(self.device)
        res = res.new_ones((dist.shape[0],dist.shape[1],dist.shape[2])) - torch.exp(-dist)

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

    def gamma_jstar(self,KEY,QUERY,Wk,Wq):
        '''
        gamma_jstar
        KEY: coordinates (x,y,...) of dim (nbatch,nbpoints,n coords)
        QUERY: coordinates (x,y,...) of dim (nbatch,nbpoints,n coords)
        Wk,Wq: range matrix of dim (nbatch,nbpoints,n coords)
        '''

        # scaling coordinates
        KEY_scale = torch.einsum('bij,bij->bij',KEY , Wk)
        QUERY_scale = torch.einsum('bkj,bkj->bkj',QUERY , Wq)

        # Euclidian scaled distance matrix between points [x,y]_i, i:1->n and points [x,y]_star
        dist = torch.cdist(KEY_scale,QUERY_scale, p=2)

        # exponential variogram of variance equal to 1
        res = torch.tensor((), dtype=torch.float64).to(self.device)
        res = res.new_ones((dist.shape[0],dist.shape[1],dist.shape[2])) - torch.exp(-dist)

        # Lagrangian multiplier
        ## tensor [b,1,N]
        lm = torch.tensor((), dtype=torch.float64).to(self.device)
        lm = lm.new_zeros((res.shape[0],1,res.shape[2]))
        res = torch.cat((res,lm),1)

        #res = torch.reshape(res,(res.shape[0],res.shape[2],res.shape[1]))

        return(res)

    def k_weight(self,KEY,QUERY,Wk,Wq):
        '''
        KEY: coordinates (x,y,...) of dim (nbatch,nbpoints,n coords)
        QUERY: coordinates (x,y,...) of dim (nbatch,nbpoints,n coords)
        Wk,Wq: range matrix of dim (nbatch,nbpoints,n coords)
        Solving this optimization problem g_ij^-1 . g_jstar (w/ Lagrange multipliers) results in the kriging system
        '''

        g_ij = self.gamma_ij(KEY,Wk)

        g_jstar = self.gamma_jstar(KEY,QUERY,Wk,Wq)

        # https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html#torch.linalg.lstsq
        #rem: torch.linalg.solve gives double
        res = torch.linalg.lstsq(g_ij,g_jstar).solution.float()

        return(res)

    def krig_pred(self,KEY,VALUE,QUERY,Wk,Wq):
        self.k_w = self.k_weight(KEY,QUERY,Wk,Wq)[:,range(KEY.shape[1])]
        res = torch.einsum('bij,bik->bjk',self.k_w,VALUE)
        return(res)
