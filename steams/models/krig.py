import torch
from steams.krig.krig import class_krig

class krignn(torch.nn.Module):

    def __init__(self,device,input_size, hidden_size):
        super(krignn, self).__init__()

        self.krig = class_krig(device)

        self.W = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, input_size))

        #self.Wq = torch.nn.Sequential(
        #    torch.nn.Linear(input_size, hidden_size),
        #    torch.nn.ReLU(),
        #    torch.nn.Linear(hidden_size, hidden_size),
        #    torch.nn.ReLU(),
        #    torch.nn.Linear(hidden_size, input_size))

    def forward(self,KEY,VALUE,QUERY):

        Wk = self.W(KEY)
        Wq = self.W(QUERY)

        res = self.krig.krig_pred(KEY,VALUE,QUERY,Wk,Wq)

        return res

class krignn2(torch.nn.Module):

    def __init__(self,device,input_size, input_v, hidden_size):
        super(krignn2, self).__init__()

        self.krig = class_krig(device)

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
            torch.nn.Linear(hidden_size, input_v))

    def forward(self,KEY,VALUE,QUERY):

        Wk = self.W(KEY)
        Wq = self.W(QUERY)
        Wv = self.Wv(VALUE)

        VALUE_scale = torch.einsum('bij,bij->bij',VALUE , Wv)

        res = self.krig.krig_pred(KEY,VALUE_scale,QUERY,Wk,Wq)

        return res
