import torch
from steams.krig.krig import class_krig

class skrignn(torch.nn.Module):

    def __init__(self,device,mlp_input_size, mlp_hidden_size, mlp_output_size):
        super(skrignn, self).__init__()

        self.krig = class_krig(device)

        # MLP w/ 2 hidden layers
        self.Lx = torch.nn.Sequential(
            torch.nn.Linear(mlp_input_size, mlp_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_size, mlp_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_size, mlp_output_size),
            #torch.nn.Threshold(0, 0.1)) # replace any values < 0 by 0.1
            torch.nn.Softplus(threshold =10.))
            # rem: torch.nn.Softplus(threshold =10.)) # constraint to get positive values, but still get zero values when large negative values

        # MLP w/ 2 hidden layers
        self.Ly = torch.nn.Sequential(
            torch.nn.Linear(mlp_input_size, mlp_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_size, mlp_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_size, mlp_output_size),
            #torch.nn.Threshold(0, 0.1)) # replace any values < 0 by 0.1
            torch.nn.Softplus(threshold =10.))
            # rem: torch.nn.Softplus(threshold =10.)) # constraint to get positive values, but still get zero values when large negative values

        # MLP w/ 2 hidden layers
        self.sig = torch.nn.Sequential(
            torch.nn.Linear(mlp_input_size, mlp_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_size, mlp_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_size, mlp_output_size),
            #torch.nn.Threshold(0, 0.1)) # replace any values < 0 by 0.1
            torch.nn.Softplus(threshold =10.))
            # rem: torch.nn.Softplus(threshold =10.)) # constraint to get positive values, but still get zero values when large negative values

    def get_Lx(self,feat_coord,target_coord):
        input_mlp = torch.cat((feat_coord,target_coord),1)
        #input_mlp = input_mlp.to(self.device)
        res = torch.clamp(self.Lx(input_mlp), min=0.001) # constraint for keeping strictely positive output
        #res = res.to(self.device)
        return(res)

    def get_Ly(self,feat_coord,target_coord):
        input_mlp = torch.cat((feat_coord,target_coord),1)
        #input_mlp = input_mlp.to(self.device)
        res = torch.clamp(self.Ly(input_mlp), min=0.001) # constraint for keeping strictely positive output
        #res = res.to(self.device)
        return(res)

    def get_sig(self,feat_coord,target_coord):
        input_mlp = torch.cat((feat_coord,target_coord),1)
        #input_mlp = input_mlp.to(self.device)
        res = torch.clamp(self.sig(input_mlp), min=0.001) # constraint for keeping strictely positive output
        #res = res.to(self.device)
        return(res)


    def forward(self,feat_coord,feat_values,target_coord):
        Lx = self.get_Lx(feat_coord,target_coord)
        Ly = self.get_Ly(feat_coord,target_coord)
        #Lt = self.get_Lt(feat_coord,target_coord)
        sig = self.get_sig(feat_coord,target_coord)

        res = self.krig.krig_pred(feat_coord,feat_values,target_coord,Lx,Ly,sig)

        return res
