from typing import Dict
from steams.dictionnary.dictionnary_trevpr import train_dict
from steams.dictionnary.dictionnary_metrics import optim_dict, criterion_dict
from steams.classes.class_model import class_model
from steams.classes.class_xyv_x import class_xyv_x
from steams.train_eval_pred.trevpr import loss, evaluation_by_target
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import os
import pickle
import torch

def k_fold_train_function(params:dict):
    # param init
    params_device = params['device']
    params_data=params['data']
    params_model=params['model']
    params_train=params['training_param']
    params_export_onnx=params['export_onnx']
    kfolds = params_train['kfolds']
    epochs = params_train['epochs']
    batch_size=params_train['batch_size']
    sessiondir = params['sessiondir']
    optim_name = params_train['optimizer']['name']
    optim_param = params_train['optimizer']['param']
    optim_fun = optim_dict[optim_name]
    crit_name = params_train['criterion']
    crit_fun = criterion_dict[crit_name[0]]

    # sessiondir
    if not os.path.exists(sessiondir):
        os.makedirs(sessiondir, exist_ok=True)
    resdir = os.path.join(sessiondir,"train")
    if not os.path.exists(resdir):
        os.makedirs(resdir, exist_ok=True)
    data_filename = os.path.join(resdir,'data_class_xyv_x.pkl')
    loss_filename = os.path.join(resdir,'loss.csv')
    folds_filename = os.path.join(resdir,'folds.pkl')

    # parameters for dataloader according to the device (cpu,cuda)
    if params_device['cuda'] is not None and torch.cuda.is_available():
        num_workers = params_device['cuda']['worker']
        pin_memory = True
    elif params_device['cpu'] is not None:
        num_workers = params_device['cpu']['worker']
        pin_memory = False
    else:
        num_workers = 0
        pin_memory = False

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=kfolds, shuffle=False) # False to keep a space and time logic
    k_folds = open(os.path.join(resdir,"kfold.pkl"), 'wb')
    pickle.dump(kfold, k_folds)

    data = class_xyv_x(params_data)
    loss_res = pd.DataFrame(columns=['fold', 'epoch', 'batch_size'])
    folds_index = []

    # loop on the folds
    for fold, (train_index, valid_index) in enumerate(kfold.split(data)):

        model_ = class_model(params_device)
        model_.build(params_model)
        train_model = train_dict[model_.name]
        train_fold  = class_xyv_x(params_data,train_index)
        valid_fold  = class_xyv_x(params_data,valid_index)
        folds_index.append({'fold': fold, 'train_index': train_index, 'valid_index': valid_index})
        f_folds = open(os.path.join(resdir,"fold_"+str(fold)+".pkl"), 'wb')
        pickle.dump(folds_index, f_folds)
        train_fold.scale(True)
        valid_fold.scale(True)
        train_data_loader = DataLoader(train_fold,batch_size=batch_size, shuffle=False,sampler=None,batch_sampler=None,num_workers=num_workers,pin_memory=pin_memory)
        valid_data_loader = DataLoader(valid_fold,batch_size=batch_size, shuffle=False,sampler=None,batch_sampler=None,num_workers=num_workers,pin_memory=pin_memory)
        opt = optim_fun(model_.model.parameters(), **optim_param)
        criterion = crit_fun()

        min_val_loss = np.Inf
        n_epochs_stop = 6
        epochs_no_improve = 0

        for epoch in range(epochs):
            res = pd.DataFrame(columns=['fold', 'epoch', 'batch_size'])
            res = res.append({'fold': fold, 'epoch': epoch, 'batch_size':batch_size},ignore_index=True)
            loss_train_tmp = pd.DataFrame(columns=crit_name)
            loss_train_tmp.loc[0,crit_name[0]] = train_model(model_, train_data_loader,opt,criterion)
            loss_valid_tmp = loss(model_, params_train, valid_data_loader)
            loss_train_tmp = loss_train_tmp.add_prefix('train_')
            loss_valid_tmp = loss_valid_tmp.add_prefix('valid_')
            res = pd.concat([res,loss_train_tmp,loss_valid_tmp],axis=1)
            loss_res = pd.concat([loss_res,res],axis=0)
            if (epoch % 5 == 0):
                model_.saveCheckpoint(resdir,"fold_"+str(fold)+"_"+str(epoch),epoch,opt,loss_train_tmp,train_index)
                loss_res.to_csv(os.path.join(resdir,"fold_"+str(fold)+"_"+str(epoch)+'loss.csv'))

            # early stopping
            if loss_valid_tmp.loc[0,'valid_'+crit_name[0]] < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = loss_valid_tmp.loc[0,'valid_'+crit_name[0]]
            else:
                epochs_no_improve += 1
            if (epoch > 5) and (epochs_no_improve == n_epochs_stop):
                print('Early stopping!' )
                break
            else:
                continue

        model_.save(resdir,"fold_"+str(fold))
        model_.export_onnx(resdir,"fold_"+str(fold),train_fold,params_export_onnx)

    # save the class_ts_x data
    f_data = open(data_filename, 'wb')
    pickle.dump(data, f_data)

    # Save the train validation history for further visualization
    f_folds = open(folds_filename, 'wb')
    pickle.dump(folds_index, f_folds)
    loss_res.to_csv(loss_filename)

def k_fold_eval_function(params:dict):
    params_sessiondir = params['sessiondir']
    params_device = params['device']
    params_data=params['data']
    params_train=params['training_param']
    params_eval=params['eval_param']
    kfolds = params_train['kfolds']
    batch_size=params_eval['batch_size']
    sessiondir = params_sessiondir['path']
    if not os.path.exists(sessiondir):
        os.makedirs(sessiondir, exist_ok=True)
    resdir = os.path.join(sessiondir,"eval")
    if not os.path.exists(resdir):
        os.makedirs(resdir, exist_ok=True)
    # parameters for dataloader according to the device (cpu,cuda)
    if params_device['cuda'] is not None and torch.cuda.is_available():
        num_workers = params_device['cuda']['worker']
        pin_memory = True
    elif params_device['cpu'] is not None:
        num_workers = params_device['cpu']['worker']
        pin_memory = False
    else:
        num_workers = 0
        pin_memory = False
    data = class_xyv_x(params_data)
    params["eval_param"]["batch_size"] = len(data)
    folds_filename = os.path.join(sessiondir,'train','folds.pkl')
    f = open(folds_filename, 'rb')
    folds_index = pickle.load(f)
    for fold in folds_index:
        fold_i = folds_index.index(fold)
        print('\n Fold: ',str(fold_i))
        metrics_filename = os.path.join(resdir,'metrics'+"_fold_"+str(fold_i)+'.csv')
        metrics_res = pd.DataFrame(columns=['fold', 'batch_size'])
        model_ = class_model(params_device)
        model_.load(os.path.join(sessiondir,"train"),"fold_"+str(fold_i))
        res = pd.DataFrame(columns=['fold', 'batch_size'])
        for i in range(len(params_data['target']['values'])):
            res = res.append({'fold': fold_i, 'batch_size':batch_size},ignore_index=True)
        res = pd.concat([res,pd.DataFrame(data={'values': params_data['target']['values']})],axis=1)
        valid_index = fold.get('valid_index')
        valid_fold  = class_xyv_x(params_data,valid_index)
        valid_fold.scale(True)
        valid_data_loader = DataLoader(valid_fold,batch_size=batch_size, shuffle=False,sampler=None,batch_sampler=None,num_workers=num_workers,pin_memory=pin_memory)
        eval_valid_tmp = evaluation_by_target(model_,  params_eval, valid_data_loader,valid_fold)
        res = pd.concat([res,eval_valid_tmp],axis=1)
        metrics_res = pd.concat([metrics_res,res],axis=0)
        metrics_res.to_csv(metrics_filename)
