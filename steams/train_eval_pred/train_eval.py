from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os

def train(obj,train_class_data,valid_class_data,niter,n_iter_stop,batch_size, shuffle=False,num_workers=0,pin_memory=False,resdir=None):
    """
    Args:
        obj: steams object
        train_class_data
        valid_class_data
        niter
        n_iter_stop
        batch_size=batch_size
        shuffle=False
        sampler=None
        batch_sampler=None
        num_workers=0
        pin_memory=False
        resdir
    """

    train_data_loader = DataLoader(train_class_data,batch_size=batch_size, shuffle=shuffle,sampler=None,batch_sampler=None,num_workers=num_workers,pin_memory=pin_memory)
    valid_data_loader = DataLoader(valid_class_data,batch_size=batch_size, shuffle=shuffle,sampler=None,batch_sampler=None,num_workers=num_workers,pin_memory=pin_memory)

    min_val_loss = np.Inf
    n_epochs_stop = n_iter_stop
    epochs_no_improve = 0

    loss_res = pd.DataFrame(columns=['epoch','train','valid'])

    for epoch in range(niter):

        loss_tmp = pd.DataFrame(columns=['epoch','train','valid'])
        loss_tmp.loc[0,'epoch'] = epoch
        loss_tmp.loc[0,'train'] = obj.single_train(train_data_loader)
        loss_tmp.loc[0,'valid'] = obj.loss(valid_data_loader)
        loss_res = pd.concat([loss_res,loss_tmp],axis=0)

        if (epoch % 5 == 0 and resdir is not None):
            obj.saveCheckpoint(resdir,str(epoch),epoch,loss_res,train_class_data.indice)
            loss_res.to_csv(os.path.join(resdir,str(epoch)+'_loss.csv'))
        elif (resdir is None):
            print(loss_tmp)

        # early stopping
        if loss_tmp.loc[0,'valid'] < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = loss_tmp.loc[0,'valid']
        else:
            epochs_no_improve += 1
        if (epoch > 5) and (epochs_no_improve == n_epochs_stop):
            print('--> early stopping' )
            break
        else:
            continue

    if (resdir is not None):
        obj.save_model(resdir,"")
    #model_.export_onnx(resdir,"fold_"+str(fold),train_fold,params_export_onnx)


def eval(obj,valid_class_data,batch_size, shuffle=False,num_workers=0,pin_memory=False,resdir=None):
    """
    Args:
        obj: steams object
        valid_class_data
        batch_size
        shuffle=False
        sampler=None
        batch_sampler=None
        num_workers=0
        pin_memory=False
        resdir
    """

    valid_data_loader = DataLoader(valid_class_data,batch_size=batch_size, shuffle=shuffle,sampler=None,batch_sampler=None,num_workers=num_workers,pin_memory=pin_memory)

    eval_tmp= obj.evaluation(valid_data_loader,valid_class_data)
    if (resdir  is not None):
        eval_tmp.to_csv(os.path.join(resdir,'eval.csv'))
    else:
        print(eval_tmp)
