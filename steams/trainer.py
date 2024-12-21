import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os


def train(obj,train_class_data,valid_class_data,niter,n_iter_stop,batch_size, shuffle=False,num_workers=0,pin_memory=False,resdir=None):
    '''
    The function train the model architecture. Metrics are processed on scaled values.

    Args:
        obj:
        BaseTrainer object that embeds the model architecture, the optimizer and its criterion, and the scheduler (if any).

        train_class_data:
        Data loader related to the data sampler for the training phase.

        valid_class_data:
        Data loader related to the data sampler for the validation phase.

        niter:
        Number of iteration of single training, i.e number of epoch.

        n_iter_stop:
        Number of epoch that determine of an early exit if no loss improvement occurs with the validation dataset.

        batch_size:
        Size of the batch

        shuffle:
        Boolean to decide whether or not to shuffle the batch, 'False' by default.

        num_workers:
        Number of workers; '0' by default.

        pin_memory:
        Pin memory, 'False' by default

        resdir:
        Path of the directory for any results to be saved, 'None' by default.
    '''

    train_class_data.scale(True)
    valid_class_data.scale(True)

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

    train_class_data.scale(False)
    valid_class_data.scale(False)

    if (resdir is not None):
        obj.save_model(resdir,"")
    #model_.export_onnx(resdir,"fold_"+str(fold),train_fold,params_export_onnx)

def evaluation(obj,valid_class_data,batch_size, shuffle=False,num_workers=0,pin_memory=False,resdir=None):
    '''
    The function evaluate the model architecture. Metrics are processed on unscaled values.

    Args:
        obj:
        BaseTrainer object that embeds the model architecture, the optimizer and its criterion, and the scheduler (if any).

        valid_class_data:
        Data loader related to the data sampler for the validation phase.

        batch_size:
        Size of the batch

        shuffle:
        Boolean to decide whether or not to shuffle the batch, 'False' by default.

        num_workers:
        Number of workers; '0' by default.

        pin_memory:
        Pin memory, 'False' by default

        resdir:
        Path of the directory for any results to be saved, 'None' by default.
    '''

    valid_class_data.scale(True)

    valid_data_loader = DataLoader(valid_class_data,batch_size=batch_size, shuffle=shuffle,sampler=None,batch_sampler=None,num_workers=num_workers,pin_memory=pin_memory)

    eval_tmp= obj.evaluation(valid_data_loader,valid_class_data)

    valid_class_data.scale(False)

    if (resdir  is not None):
        eval_tmp.to_csv(os.path.join(resdir,'eval.csv'))
    else:
        print(eval_tmp)

def prediction_prime(obj,class_data,resdir=None):
    '''
    The function infer the model architecture and concatenate the results with observations.

    Args:
        obj:
        BaseTrainer object that embeds the model architecture, the optimizer and its criterion, and the scheduler (if any).

        class_data:
        KVyQVx data sampler used to extract scale parameters.

        resdir:
        Path of the directory for any results to be saved, 'None' by default.
    '''

    class_data.scale(True)

    nrow = len(class_data)
    pred_name = ['pred_' + v for v in class_data.VALUE_X]
    results = pd.DataFrame(columns=class_data.QUERY+class_data.VALUE_X+pred_name)
    for i in range(1,nrow,1):
        KEY_Y, VALUE_Y, QUERY_X, VALUE_X = class_data[i]

        QUERY_X_unscaled, VALUE_X_pred_unscaled = obj.predict(KEY_Y,VALUE_Y,QUERY_X,class_data)

        #unscale VALUE_X
        VALUE_X_unscaled = class_data.unscale(VALUE_X.detach(),"VALUE_X").to(obj.device)

        QUERY_X_unscaled = torch.reshape(QUERY_X_unscaled,(QUERY_X_unscaled.shape[1],QUERY_X_unscaled.shape[2]))
        VALUE_X_pred_unscaled = torch.reshape(VALUE_X_pred_unscaled,(VALUE_X_pred_unscaled.shape[1],VALUE_X_pred_unscaled.shape[2]))

        tmp = torch.concat((QUERY_X_unscaled, VALUE_X_unscaled,VALUE_X_pred_unscaled),1).cpu().numpy()
        tmp_df = pd.DataFrame(tmp,columns=class_data.QUERY+class_data.VALUE_X+pred_name)

        results = pd.concat([results,tmp_df],ignore_index=True,axis=0)

    class_data.scale(False)

    if (resdir  is not None):
        results.to_csv(os.path.join(resdir,'prediction_prime.csv'))
    else:
        return results

def ensemble_prime(obj,class_data,N,q=[0.05, 0.5, 0.95],resdir=None):
    '''
    The function creates an ensemble of predictions, and provides values related to a percentile of the member, and the p-value of each observation within the members.

    Args:
        obj:
        BaseTrainer object that embeds the model architecture, the optimizer and its criterion, and the scheduler (if any).

        class_data:
        KVyQVx data sampler used to extract scale parameters.

        N:
        Number of times to repeat the inference.

        q:
        List of percentiles; by default the 5-percentile, 50-percentile and 95-percentile.

        resdir:
        Path of the directory for any results to be saved, 'None' by default.
    '''

    pred_name = ['pred_' + v for v in class_data.VALUE_X]
    results = pd.DataFrame(columns=class_data.QUERY+class_data.VALUE_X+pred_name)
    for n in range(1,N,1):
        results = pd.concat([results,prediction_prime(obj,class_data,resdir=None)],ignore_index=True,axis=0)

    for v in class_data.VALUE_X:
        results.loc[:,'p_'+v] = results.loc[:,v]<=results.loc[:,'pred_'+v]

    # quantile of the predictions
    ensemble_quantile = results.groupby(class_data.QUERY)[pred_name].agg([lambda x: np.quantile(x,q=0.05), lambda x: np.quantile(x,q=0.5),lambda x: np.quantile(x,q=0.95)]).rename(
        columns={"<lambda_0>": "q0_05", "<lambda_1>": "q0_5", "<lambda_2>": "q0_95"})

    # observations
    ensemble_observations = results.groupby(class_data.QUERY)[class_data.VALUE_X].agg([np.size,np.mean])

    # p_value of VALUE_X among the predictions
    p_name = ['p_' + v for v in class_data.VALUE_X]
    ensemble_pvalue =results.groupby(class_data.QUERY)[p_name].aggregate(np.mean)

    ensemble = pd.concat([ensemble_observations,ensemble_pvalue,ensemble_quantile,],axis=1)

    ensemble = ensemble.reset_index()

    if (resdir  is not None):
        ensemble.to_csv(os.path.join(resdir,'ensemble.csv'))
    else:
        return ensemble

class BaseTrainer():
    def __init__(self,model,device):
        '''
        BaseTrainer enables to gather optimizer, scheduler, criterion in one object and simplify the design of function train, loss, evaluation and prediction.

        Args:
            model:
            Model of class torch.nn.Module

            device:
            Determined with torch.device()
        '''
        self.device = device
        self.model = model
        self.model.to(self.device)

    def init_optimizer(self,optimizer):
        self.optimizer = optimizer

    def init_criterion(self,criterion):
        self.criterion = criterion

    def saveCheckpoint(self, path: str, name: str, epoch, loss, index=None):
        os.makedirs(path, exist_ok=True)  # Ensures the directory exists
        checkpoint_files = [f for f in os.listdir(path) if f.endswith('_checkpoint.pth')]
        if len(checkpoint_files) >= 10:  # Adjusted to prevent errors
            os.remove(os.path.join(path, checkpoint_files[0]))  # Removes the oldest checkpoint
        checkpoint_path = os.path.join(path, f"{name}_checkpoint.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'index': index
        }, checkpoint_path)

    def save_model(self, path: str, name:str) -> None:
        if not os.path.exists(path):
            os.mkdir(path)
        model_path = os.path.join(path, name + "_model.pth")
        torch.save(self.model.state_dict(), model_path)

    def load_model(self,path:str, name:str) -> None:
        model_path = os.path.join(path, name + "_model.pth")
        self.model.load_state_dict(torch.load(model_path))

class AttentionTrainer(BaseTrainer):
    def __init__(self,model,device):
        '''
        attention_steams provide function train, loss, evaluation and prediction for additive attention.

        Args:
            model:
            Model of class torch.nn.Module

            device:
            Determined with torch.device()
        '''

        super(AttentionTrainer, self).__init__(model,device)

    def single_train(self,data_loader):
        '''
        The function trains the model architecture for one epoch.

        Args:
            data_loader:
            Data loader based on the KVyQVx data sampler.
        '''
        running_loss = 0.0
        self.model.train()
        for i, (KEY_Y,VALUE_Y,QUERY_X,VALUE_X) in enumerate(data_loader):

            KEY_Y = KEY_Y.to(self.device)
            VALUE_Y = VALUE_Y.to(self.device)
            QUERY_X = QUERY_X.to(self.device)
            VALUE_X = VALUE_X.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(KEY_Y.float() ,VALUE_Y.float() ,QUERY_X.float() )
            loss_ = self.criterion(VALUE_X.float(),output)
            loss_.backward()
            self.optimizer.step()
            #self.scheduler_lr.step()

            if torch.isnan(loss_) or loss_ == float('inf'):
                raise ValueError("Error: Infinite or NaN loss detected.")
            running_loss += loss_.item()
        avg_loss = running_loss / (float(i)+1.)
        return avg_loss

    def loss(self,data_loader):
        '''
        The function provides the loss for one epoch. It provides metrics for scaled data.

        Args:
            data_loader:
            Data loader based on the KVyQVx data sampler.
        '''
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for i, (KEY_Y,VALUE_Y,QUERY_X,VALUE_X) in enumerate(data_loader):

                KEY_Y = KEY_Y.to(self.device)
                VALUE_Y = VALUE_Y.to(self.device)
                QUERY_X = QUERY_X.to(self.device)
                VALUE_X = VALUE_X.to(self.device)

                output = self.model(KEY_Y.float(),VALUE_Y.float(), QUERY_X.float())
                loss_ = self.criterion(VALUE_X.float(),output)
                running_loss += loss_.item()
            avg_loss = running_loss / (float(i)+1.)
        return avg_loss

    def evaluation(self, data_loader, class_data):
        '''
        The function evaluates the model architecture for one epoch. It provides metrics for unscaled data.

        Args:
            data_loader:
            Data loader based on the KVyQVx data sampler.

            class_data:
            KVyQVx data sampler used to extract scale parameters.
        '''
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for i, (KEY_Y,VALUE_Y,QUERY_X,VALUE_X) in enumerate(data_loader):
                KEY_Y = KEY_Y.to(self.device)
                VALUE_Y = VALUE_Y.to(self.device)
                QUERY_X = QUERY_X.to(self.device)
                VALUE_X = VALUE_X.to(self.device)

                output = self.model(KEY_Y.float(),VALUE_Y.float(), QUERY_X.float())

                #unscale
                output_unscale = class_data.unscale(output,"VALUE_X").to(self.device)
                VALUE_X_unscale = class_data.unscale(VALUE_X,"VALUE_X").to(self.device)

                loss_ = self.criterion( VALUE_X_unscale.float(),output_unscale)
                running_loss += loss_.item()

            avg_loss = running_loss / (float(i)+1.)
        return avg_loss

    def predict(self, KEY_Y,VALUE_Y,QUERY_X,class_data):
        '''
        The function infers the model arhcitecture. The output are unscaled.

        Args:
            KEY_Y:
            Tensor Key of dimension (nb_points,input_k)

            VALUE_Y:
            Tensor Values related to the keys of dimension (nb_points,input_v)

            QUERY_X:
            Tensor Query of dimension (nb_points,input_q)

            class_data:
            KVyQVx data sampler used to extract scale parameters.
        '''
        self.model.eval()
        with torch.no_grad():

            # Input reshaping for batch inference
            KEY_Y = KEY_Y.to(self.device).unsqueeze(0)
            VALUE_Y = VALUE_Y.to(self.device).unsqueeze(0)
            QUERY_X = QUERY_X.to(self.device).unsqueeze(0)

            VALUE_X_pred = self.model(KEY_Y.float() ,VALUE_Y.float() ,QUERY_X.float() ).detach()

            VALUE_X_pred_unscaled = class_data.unscale(VALUE_X_pred,"VALUE_X").to(self.device)
            QUERY_X_unscaled = class_data.unscale(QUERY_X.detach(),"QUERY").to(self.device)

        return QUERY_X_unscaled, VALUE_X_pred_unscaled
