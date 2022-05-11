from typing import Dict
from steams.dictionnary.dictionnary_trevpr import predict_dict, loss_dict, eval_bytarget_dict
from steams.dictionnary.dictionnary_metrics import optim_dict, criterion_dict
from steams.classes.class_model import class_model
from torch.utils.data import DataLoader
import pandas as pd

def loss(model: class_model, params_eval:dict, data_loader: DataLoader):
    loss_model = loss_dict[model.name]
    crit_name = params_eval['criterion']
    res = pd.DataFrame(columns=params_eval['criterion'])
    for i in range(len(crit_name)):
        crit_fun = criterion_dict[crit_name[i]]
        res.loc[0,crit_name[i]] = loss_model(model, data_loader, crit_fun)
    return(res)

def evaluation_by_target(model: class_model, params_eval:dict, data_loader: DataLoader, class_ts_x):
    evaluation_bytarget_model = eval_bytarget_dict[model.name]
    crit_name = params_eval['criterion']
    res = pd.DataFrame()
    for i in range(len(crit_name)):
        crit_fun = criterion_dict[crit_name[i]]
        res.loc[:,crit_name[i]] = evaluation_bytarget_model(model, data_loader, crit_fun, class_ts_x)['crit']
    return(res)

def predict(model: class_model, class_ts_):
    predict_model = predict_dict[model.name]
    res = predict_model(model, class_ts_)
    return(res)
