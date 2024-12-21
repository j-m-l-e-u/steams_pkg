import math
import torch

class RMSE(torch.nn.Module):
    def __init__(self):
        '''
        Root Mean Square Error
        '''
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self,target: torch.Tensor,y_pred: torch.Tensor):
        loss = torch.sqrt(self.mse(target,y_pred))
        return loss

class MAPE(torch.nn.Module):
    def __init__(self):
        '''
        Mean Absolute Percentage Error
        '''
        super().__init__()
    def forward(self, target: torch.Tensor, y_pred: torch.Tensor):
        return torch.mean(torch.abs((target - y_pred) / target))

class bias(torch.nn.Module):
    '''
    Bias
    '''
    def __init__(self):
        super().__init__()
    def forward(self, target: torch.Tensor, y_pred: torch.Tensor):
        bias = torch.mean(y_pred-target)
        return bias

class variance(torch.nn.Module):
    '''
    Variance
    '''
    def __init__(self):
        super().__init__()
    def forward(self, target: torch.Tensor, y_pred: torch.Tensor):
        bias = torch.mean(y_pred-target)
        res = torch.mean(torch.abs(y_pred - target - bias))**2
        return res

# adapted from https://pytorch-widedeep.readthedocs.io/en/latest/_modules/pytorch_widedeep/metrics.html#R2Score
class R2(torch.nn.Module):
    '''
    Coefficient of determination
    '''
    def __init__(self):
        super().__init__()
        self.numerator = 0
        self.denominator = 0
        self.num_examples = 0
        self.y_true_sum = 0
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        self.numerator += ((y_pred - y_true) ** 2).sum()
        self.num_examples += torch.prod(torch.tensor(y_true.shape))
        self.y_true_sum += y_true.sum()
        y_true_avg = self.y_true_sum / self.num_examples
        self.denominator += ((y_true - y_true_avg) ** 2).sum()
        res = 1 - (self.numerator / self.denominator)
        return res
