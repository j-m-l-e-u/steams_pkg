import torch
from steams.krig.krig import class_krig

class skrignn(torch.nn.Module):

    def __init__(self,device,mlp_input_size, mlp_hidden_size, mlp_output_size):
        super(skrignn, self).__init__()

        self.krig = class_krig(device)

        # MLP w/ 1 hidden layers
        self.Lx = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size))
            #torch.nn.Linear(mlp_hidden_size, mlp_hidden_size),
            #torch.nn.ReLU(),
            #torch.nn.Linear(mlp_hidden_size, mlp_output_size),
            ##torch.nn.Threshold(0, 0.1)) # replace any values < 0 by 0.1
            #torch.nn.Softplus(threshold =10.))
            # rem: torch.nn.Softplus(threshold =10.)) # constraint to get positive values, but still get zero values when large negative values

        # MLP w/ 1 hidden layers
        self.Ly = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size))
            #torch.nn.Linear(mlp_hidden_size, mlp_hidden_size),
            #torch.nn.ReLU(),
            #torch.nn.Linear(mlp_hidden_size, mlp_output_size),
            ##torch.nn.Threshold(0, 0.1)) # replace any values < 0 by 0.1
            #torch.nn.Softplus(threshold =10.))
            # rem: torch.nn.Softplus(threshold =10.)) # constraint to get positive values, but still get zero values when large negative values

        # MLP w/ 1 hidden layers
        self.sig = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size))
            #torch.nn.Linear(mlp_hidden_size, mlp_hidden_size),
            #torch.nn.ReLU(),
            #torch.nn.Linear(mlp_hidden_size, mlp_output_size),
            #torch.nn.Threshold(0, 0.1)) # replace any values < 0 by 0.1
            #torch.nn.Softplus(threshold =10.))
            # rem: torch.nn.Softplus(threshold =10.)) # constraint to get positive values, but still get zero values when large negative values

    def get_Lx(self,KEY,QUERY):
        input_mlp = torch.cat((KEY,QUERY),1)
        res = torch.clamp(self.Lx(input_mlp), min=0.001) # constraint for keeping strictely positive output
        return(res)

    def get_Ly(self,KEY,QUERY):
        input_mlp = torch.cat((KEY,QUERY),1)
        res = torch.clamp(self.Ly(input_mlp), min=0.001) # constraint for keeping strictely positive output
        return(res)

    def get_sig(self,KEY,QUERY):
        input_mlp = torch.cat((KEY,QUERY),1)
        res = torch.clamp(self.sig(input_mlp), min=0.001) # constraint for keeping strictely positive output
        return(res)


    def forward(self,KEY,VALUE,QUERY):
        Lx = self.get_Lx(KEY,QUERY)
        Ly = self.get_Ly(KEY,QUERY)
        sig = self.get_sig(KEY,QUERY)

        res = self.krig.krig_pred(KEY,VALUE,QUERY,Lx,Ly,sig)

        return res
