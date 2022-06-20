import torch

class class_krig():
    def __init__(self,device):
        self.device = device

    def get_dist_ij(self,feat_coord,Lx,Ly):
        # scaled coordinates
        Lx_Ly = torch.cat((Lx,Ly),2)
        coords_f_scaled = torch.div(feat_coord,  Lx_Ly[:,0:feat_coord.shape[1],:])
        res = torch.cdist(coords_f_scaled,coords_f_scaled, p=2)
        res = res.to(self.device)
        return(res)

    def gamma_ij(self,feat_coord,Lx,Ly,sig):
        '''
        gamma_ij
        feat_coord: coordinates (x,y) of dim (nbatch,nbpoints,2)
        Lx,Ly,sig: range matrix of size nbatch x nbPoint x 1
        '''

        # scaled coordinates
        Lx_Ly = torch.cat((Lx,Ly),2)
        coords_f_scaled = torch.div(feat_coord[:,:,range(2)],  Lx_Ly[:,0:feat_coord.shape[1],:])

        # Euclidian scaled distance matrix between points [x,y]_i, i:1->n and points [x,y]_j, j:1->n
        dist = torch.cdist(coords_f_scaled,coords_f_scaled, p=2)
        dist = dist.to(self.device)

        # variance
        variance = torch.mul(sig[:,0:feat_coord.shape[1],:],torch.transpose(sig[:,0:feat_coord.shape[1],:],1,2))

        # exponential variogram
        res = torch.tensor((), dtype=torch.float64).to(self.device)
        res = res.new_ones((dist.shape[0],dist.shape[1],dist.shape[2])) - torch.exp(-dist)
        res = torch.mul(variance,res)

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

    def gamma_jstar(self,feat_coord,target_coord,Lx,Ly,sig):
        '''
        gamma_jstar
        feat_coord: coordinates (x,y) of dim (nbatch,nbpoints,2)
        target_coord: oordinates (x,y) of dim (nbatch,nbpoints,2)
        Lx,Ly,sig: range matrix of size nbatch x nbPoint x 1
        '''
        # scaled coordinates
        Lx_Ly = torch.cat((Lx,Ly),2)
        coords_f_scaled = torch.div(feat_coord[:,:,range(2)],  Lx_Ly[:,0:feat_coord.shape[1],:])
        coords_t_scaled = torch.div(target_coord[:,:,range(2)],  Lx_Ly[:,feat_coord.shape[1]:(feat_coord.shape[1]+target_coord.shape[1]),:])

        # Euclidian scaled distance matrix between points [x,y]_i, i:1->n and points [x,y]_star
        dist = torch.cdist(coords_f_scaled,coords_t_scaled, p=2)
        #dist = dist.to(self.device)

        # variance
        variance = torch.mul(sig[:,0:feat_coord.shape[1],:],torch.transpose(sig[:,feat_coord.shape[1]:(feat_coord.shape[1]+target_coord.shape[1]),:],1,2))

        # exponential variogram
        res = torch.tensor((), dtype=torch.float64).to(self.device)
        res = res.new_ones((dist.shape[0],dist.shape[1],dist.shape[2])) - torch.exp(-dist)
        res = torch.mul(variance,res)

        # Lagrangian multiplier
        ## tensor [b,1,N]
        lm = torch.tensor((), dtype=torch.float64).to(self.device)
        lm = lm.new_zeros((res.shape[0],1,res.shape[2]))
        res = torch.cat((res,lm),1)

        #res = torch.reshape(res,(res.shape[0],res.shape[2],res.shape[1]))

        return(res)

    def k_weight(self,feat_coord,target_coord,Lx,Ly,sig):
        '''
        feat_coord: coordinates (x,y) of dim (nbatch,nbpoints,2)
        target_coord: oordinates (x,y) of dim (nbatch,nbpoints,2)
        L_L_sig: range matrix of size nbatch x nbPoint x 3 (Lx,Ly,sigma)
        Solving this optimization problem g_ij^-1 . g_jstar (w/ Lagrange multipliers) results in the kriging system
        rem: torch.linalg.solve gives double
        '''

        g_ij = self.gamma_ij(feat_coord,Lx,Ly,sig)

        g_jstar = self.gamma_jstar(feat_coord,target_coord,Lx,Ly,sig)

        # https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html#torch.linalg.lstsq
        res = torch.linalg.lstsq(g_ij,g_jstar).solution.float()

        return(res)

    def krig_pred(self,feat_coord,feat_values,target_coord,Lx,Ly,sig):

        k_w = self.k_weight(feat_coord,target_coord,Lx,Ly,sig)[:,range(feat_coord.shape[1])]

        res = torch.sum(torch.mul(k_w,feat_values),dim=(1),keepdim=True)
        res = res.permute(0,2,1) 
        return(res)
