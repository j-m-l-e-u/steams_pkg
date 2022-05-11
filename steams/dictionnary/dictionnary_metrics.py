
from torch.optim import Adam, SGD
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
from steams.utils.criterion import RMSE, MAPE, R2,bias

optim_dict = {"adam": Adam, "sgd": SGD}

criterion_dict = {
    "mse": MSELoss,
    "smooth_l1_loss": SmoothL1Loss,
    "rmse": RMSE,
    "mape": MAPE,
    "l1": L1Loss,
    "r2": R2,
    "bias": bias}

#NSE, KGE,...
