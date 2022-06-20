from typing import Dict
from steams.dict.int import steams_dict,scheduler_dict
from steams.dict.ext import optim_dict, criterion_dict
from steams.data.class_xyv_x import class_xyv_x
from steams.utils.model import build_model, load_model
from steams.train_eval_pred.train_eval import train, eval
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import os
import pickle
import torch

def k_fold_train_function(params:dict):

    ################
    # param init   #
    ################
    params_device = params['device']
    params_data=params['data']
    params_model=params['model']
    params_train=params['training_param']
    params_export_onnx=params['export_onnx']
    kfolds = params_train['kfolds']
    epochs = params_train['epochs']
    n_iter_stop = params_train['epochs_no_impr']
    batch_size=params_train['batch_size']
    sessiondir = params['sessiondir']

    optim_name = params_train['optimizer']['name']
    optim_param = params_train['optimizer']['param']
    optim_fun = optim_dict[optim_name]

    crit_name = params_train['criterion']
    crit_fun = criterion_dict[crit_name]

    scheduler_fun = scheduler_dict[params_train['scheduler']['name']]
    scheduler_param = params_train['scheduler']['param']

    class_steams = steams_dict[params_model['name']]

    ###############
    # sessiondir  #
    ###############
    if not os.path.exists(sessiondir):
        os.makedirs(sessiondir, exist_ok=True)
    resdir = os.path.join(sessiondir,"train")
    if not os.path.exists(resdir):
        os.makedirs(resdir, exist_ok=True)

    #################################################################
    # parameters for dataloader according to the device (cpu,cuda)  #
    #################################################################
    if params_device['cuda'] is not None and torch.cuda.is_available():
        num_workers = params_device['cuda']['worker']
        cuda_name = params_device['cuda']['name']
        pin_memory = True
        device = torch.device('cuda'+":"+cuda_name)
    elif params_device['cpu'] is not None:
        num_workers = params_device['cpu']['worker']
        pin_memory = False
        device = torch.device('cpu')
    else:
        num_workers = 0
        pin_memory = False
        device = torch.device('cpu')

    ######################################
    # Define the K-fold Cross Validator  #
    ######################################
    data = class_xyv_x(params_data)
    kfold = KFold(n_splits=kfolds, shuffle=False) # False to keep a space and time logic
    folds_index = []
    f_folds_index = open(os.path.join(resdir,"fold_index.pkl"), 'wb')

    ######################
    # loop on the folds  #
    ######################
    for fold, (train_index, valid_index) in enumerate(kfold.split(data)):

        folds_index.append({'fold': fold, 'train_index': train_index, 'valid_index': valid_index})
        pickle.dump(folds_index, f_folds_index)

        # build data generators
        train_fold  = class_xyv_x(params_data,train_index)
        valid_fold  = class_xyv_x(params_data,valid_index)

        # create folder fold_x to save results of the training of the fold
        os.makedirs(os.path.join(resdir,'fold_'+str(fold)), exist_ok=True)

        # build nn model
        model = build_model(params_model)

        # optimzer
        optimizer = optim_fun(model.parameters(), **optim_param)

        # criterion
        criterion = crit_fun()

        #scheduler
        scheduler = scheduler_fun(optimizer,scheduler_param)

        # train object
        obj = class_steams(model,device)
        obj.init_optimizer(optimizer)
        obj.init_criterion(criterion)
        obj.init_scheduler_lr(scheduler)

        # scaling
        train_fold.scale(True)
        valid_fold.scale(True)

        # training
        train(obj,train_fold,valid_fold,niter=epochs,n_iter_stop=n_iter_stop,batch_size=batch_size,shuffle=False,sampler=None,batch_sampler=None,num_workers=num_workers,pin_memory=pin_memory,resdir=resdir)



def k_fold_eval_function(params:dict):

    ###############
    # param init  #
    ###############
    params_sessiondir = params['sessiondir']
    params_device = params['device']
    params_data=params['data']
    params_train=params['training_param']
    params_eval=params['eval_param']
    kfolds = params_train['kfolds']
    batch_size=params_eval['batch_size']
    sessiondir = params_sessiondir['path']
    crit_name = params_eval['criterion']
    class_steams = steams_dict[params_model['name']]

    ###############
    # sessiondir  #
    ###############
    if not os.path.exists(sessiondir):
        os.makedirs(sessiondir, exist_ok=True)

    #################################################################
    # parameters for dataloader according to the device (cpu,cuda)  #
    #################################################################
    if params_device['cuda'] is not None and torch.cuda.is_available():
        num_workers = params_device['cuda']['worker']
        cuda_name = params_device['cuda']['name']
        pin_memory = True
        device = torch.device('cuda'+":"+cuda_name)
    elif params_device['cpu'] is not None:
        num_workers = params_device['cpu']['worker']
        pin_memory = False
        device = torch.device('cpu')
    else:
        num_workers = 0
        pin_memory = False
        device = torch.device('cpu')

    #data = class_xyv_x(params_data)
    #params["eval_param"]["batch_size"] = len(data)

    ######################
    # loop on the folds  #
    ######################
    folds_filename = os.path.join(sessiondir,'train','fold_index.pkl')
    f = open(folds_filename, 'rb')
    folds_index = pickle.load(f)

    for fold in folds_index:

        fold_i = folds_index.index(0)
        print('\n Fold: ',str(fold_i))

        # load model
        model = load_model(os.path.join(sessiondir,"train","fold_"+str(fold_i)))
        model.eval()

        # build data generators
        valid_index = fold.get('valid_index')
        valid_fold  = class_xyv_x(params_data,valid_index)

        # loop on the evaluation criteria
        for i in range(len(crit_name)):

            criterion = criterion_dict[crit_name[i]]

            resdir = os.path.join(sessiondir,"eval",crit_name[i],'fold_'+str(fold))
            if not os.path.exists(resdir):
                os.makedirs(resdir, exist_ok=True)

            # eval object
            obj = class_steams(model,device)
            obj.init_criterion(criterion)

            # scaling
            valid_fold.scale(True)

            # evaluation
            eval(obj,valid_fold,batch_size=batch_size,shuffle=False,sampler=None,batch_sampler=None,num_workers=num_workers,pin_memory=pin_memory,resdir=resdir)
