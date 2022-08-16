import os
import torch
import pandas as pd
import numpy as np

class class_steams():
    def __init__(self,model,device):
        self.device = device
        self.model = model
        self.model.to(self.device)

    def init_optimizer(self,optimizer):
        self.optimizer = optimizer

    def init_scheduler_lr(self,scheduler_lr):
        self.scheduler_lr = scheduler_lr

    def init_criterion(self,criterion):
        self.criterion = criterion

    def saveCheckpoint(self,path: str, name:str, epoch, loss,index=None):
        if not os.path.exists(path):
            os.mkdir(path)
        checkpoint_files = [f for f in os.listdir(path) if f.endswith('_checkpoint.pth')]
        if len(checkpoint_files)==10:
            for file in checkpoint_files:
                os.remove(os.path.join(path, file))
        checkpoint_path = os.path.join(path, name + "_checkpoint.pth")
        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                    'index': index}, checkpoint_path)

    def save_model(self, path: str, name:str) -> None:
        if not os.path.exists(path):
            os.mkdir(path)
        model_path = os.path.join(path, name + "_model.pth")
        torch.save(self.model.state_dict(), model_path)

class attention_steams(class_steams):
    def __init__(self,model,device):
        super(attention_steams, self).__init__(model,device)

    def single_train(self,data_loader):
        running_loss = 0.0
        self.model.train()
        for i, (KEY_Y,VALUE_Y,QUERY_X,VALUE_X) in enumerate(data_loader):

            KEY_Y = KEY_Y.to(self.device)
            VALUE_Y = VALUE_Y.to(self.device)
            QUERY_X = QUERY_X.to(self.device)
            VALUE_X = VALUE_X.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(KEY_Y.float() ,VALUE_Y.float() ,QUERY_X.float() )

            loss = self.criterion(VALUE_X.float(),output)
            loss.backward()
            self.optimizer.step()
            #self.scheduler_lr.step()

            if torch.isnan(loss) or loss == float('inf'):
                raise("Error infinite or NaN loss detected")
            running_loss += loss.item()
        avg_loss = running_loss / (float(i)+1.)
        return avg_loss

    def loss(self,data_loader):
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for i, (KEY_Y,VALUE_Y,QUERY_X,VALUE_X) in enumerate(data_loader):

                KEY_Y = KEY_Y.to(self.device)
                VALUE_Y = VALUE_Y.to(self.device)
                QUERY_X = QUERY_X.to(self.device)
                VALUE_X = VALUE_X.to(self.device)

                output = self.model(KEY_Y.float(),VALUE_Y.float(), QUERY_X.float())
                loss = self.criterion( VALUE_X.float(),output)
                running_loss += loss.item()
            avg_loss = running_loss / (float(i)+1.)
        return avg_loss

    def evaluation_bytarget(self, data_loader, class_xyv_):
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for i, (KEY_Y,VALUE_Y,QUERY_X,VALUE_X) in enumerate(data_loader):
                KEY_Y = KEY_Y.to(self.device)
                VALUE_Y = VALUE_Y.to(self.device)
                QUERY_X = QUERY_X.to(self.device)
                VALUE_X = VALUE_X.to(self.device)

                output = self.model(KEY_Y.float(),VALUE_Y.float(), QUERY_X.float())

                ##unscale
                #output_unscale = class_xyv_.unscale(output,"values")
                #target_unscale = class_xyv_.unscale(VALUE_X,"values")

                loss = self.criterion( VALUE_X.float(),output)
                running_loss += loss.item()

            avg_loss = running_loss / (float(i)+1.)
        return avg_loss

    def predict(self, class_xyv_):
        self.model.eval()
        with torch.no_grad():
            output=[]
            for i, (KEY_Y,VALUE_Y,QUERY_X, _ ) in enumerate(class_xyv_):
                KEY_Y = KEY_Y.to(self.device)
                VALUE_Y = VALUE_Y.to(self.device)
                QUERY_X = QUERY_X.to(self.device)

                tmp = self.model(KEY_Y.float() ,VALUE_Y.float() ,QUERY_X.float() )
                output.append(tmp)
            res = torch.cat(output, axis=0).to(self.device).detach()
        return res

class transformer_ae_steams(class_steams):
    def __init__(self,model,device):
        super(transformer_ae_steams, self).__init__(model,device)

    def single_train(self, data_loader):
        running_loss = 0.0
        self.model.train()
        for i, (KEY_Y,VALUE_Y,QUERY_X,VALUE_X) in enumerate(data_loader):

            KEY_Y = KEY_Y.to(self.device)
            VALUE_Y = VALUE_Y.to(self.device)
            QUERY_X = QUERY_X.to(self.device)
            VALUE_X = VALUE_X.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(KEY_Y.float(),VALUE_Y.float())

            loss = self.criterion(VALUE_X.float(),output)
            loss.backward()
            self.optimizer.step()
            self.scheduler_lr.step()

            if torch.isnan(loss) or loss == float('inf'):
                raise("Error infinite or NaN loss detected")
            running_loss += loss.item()
        avg_loss = running_loss / float(i)
        return avg_loss

    def loss(self,data_loader):
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for i, (KEY_Y,VALUE_Y,QUERY_X,VALUE_X) in enumerate(data_loader):

                KEY_Y = KEY_Y.to(self.device)
                VALUE_Y = VALUE_Y.to(self.device)
                QUERY_X = QUERY_X.to(self.device)
                VALUE_X = VALUE_X.to(self.device)

                output = self.model(KEY_Y.float(),VALUE_Y.float())
                loss = self.criterion( VALUE_X.float(),output)
                running_loss += loss.item()
            avg_loss = running_loss / float(i)
        return avg_loss

    def evaluation_bytarget(self, data_loader, class_xyv_):
        self.model.eval()
        with torch.no_grad():
            tmp = pd.DataFrame()
            running_loss = np.zeros(2, dtype = int)
            for i, (KEY_Y,VALUE_Y,QUERY_X,VALUE_X) in enumerate(data_loader):
                KEY_Y = KEY_Y.to(self.device)
                VALUE_Y = VALUE_Y.to(self.device)
                QUERY_X = QUERY_X.to(self.device)
                VALUE_X = VALUE_X.to(self.device)

                output = self.model(KEY_Y.float(),VALUE_Y.float())

                ##unscale
                #output_unscale = class_xyv_.unscale(output,"values")
                #target_unscale = class_xyv_.unscale(VALUE_X,"values")

                for k in range(output.shape[1]):
                    #loss = self.criterion(output_unscale[:,k], target_unscale[:,k])
                    loss = self.criterion(output[:,k], VALUE_X[:,k])
                    tmp.loc[i,k] = loss.item()
            avg_loss = pd.DataFrame({'crit':tmp.apply(lambda x: np.mean(x))})
        return avg_loss

    def predict(self, class_xyv_):
        self.model.eval()
        with torch.no_grad():
            output=[]
            for i, (KEY_Y,VALUE_Y,QUERY_X, _ ) in enumerate(class_xyv_):
                KEY_Y = KEY_Y.to(self.device)
                VALUE_Y = VALUE_Y.to(self.device)
                QUERY_X = QUERY_X.to(self.device)

                tmp = self.model(KEY_Y.float(),VALUE_Y.float())
                output.append(tmp)
            res = torch.cat(output, axis=0).to(self.device).detach()
        return res

class transformer_steams(class_steams):
    def __init__(self,model,device):
        super(transformer_steams, self).__init__(model,device)

    def single_train(self, data_loader):
        running_loss = 0.0
        self.model.train()
        for i, (KEY_Y,VALUE_Y,QUERY_X,VALUE_X) in enumerate(data_loader):

            QUERY_X = QUERY_X.to(self.device) # X
            VALUE_X = VALUE_X.to(self.device) # coords X
            KEY_Y = KEY_Y.to(self.device) # Y
            VALUE_Y = VALUE_Y.to(self.device) # coords Y


            self.optimizer.zero_grad()
            output = self.model(QUERY_X.float(),VALUE_X.float(), KEY_Y.float(),VALUE_Y.float())[0]
            loss = self.criterion(VALUE_X.float(),output)
            loss.backward()
            self.optimizer.step()
            self.scheduler_lr.step()

            if torch.isnan(loss) or loss == float('inf'):
                raise("Error infinite or NaN loss detected")
            running_loss += loss.item()
        avg_loss = running_loss / float(i)
        return avg_loss

    def loss(self,data_loader):
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for i, (KEY_Y,VALUE_Y,QUERY_X,VALUE_X) in enumerate(data_loader):

                KEY_Y = KEY_Y.to(self.device)
                VALUE_Y = VALUE_Y.to(self.device)
                QUERY_X = QUERY_X.to(self.device)
                VALUE_X = VALUE_X.to(self.device)

                output = self.model(QUERY_X.float(),VALUE_X.float(), KEY_Y.float(),VALUE_Y.float())[0]
                loss = self.criterion( VALUE_X.float(),output)
                running_loss += loss.item()
            avg_loss = running_loss / float(i)
        return avg_loss

    def evaluation_bytarget(self, data_loader, class_xyv_):
        self.model.eval()
        with torch.no_grad():
            tmp = pd.DataFrame()
            running_loss = np.zeros(2, dtype = int)
            for i, (KEY_Y,VALUE_Y,QUERY_X,VALUE_X) in enumerate(data_loader):
                KEY_Y = KEY_Y.to(self.device)
                VALUE_Y = VALUE_Y.to(self.device)
                QUERY_X = QUERY_X.to(self.device)
                VALUE_X = VALUE_X.to(self.device)

                output = self.model(QUERY_X.float(),VALUE_X.float(), KEY_Y.float(),VALUE_Y.float())[0]

                ##unscale
                #output_unscale = class_xyv_.unscale(output,"values")
                #target_unscale = class_xyv_.unscale(VALUE_X,"values")

                for k in range(output.shape[1]):
                    #loss = self.criterion(output_unscale[:,k], target_unscale[:,k])
                    loss = self.criterion(output[:,k], VALUE_X[:,k])
                    tmp.loc[i,k] = loss.item()
            avg_loss = pd.DataFrame({'crit':tmp.apply(lambda x: np.mean(x))})
        return avg_loss

    def predict(self, class_xyv_):
        self.model.eval()
        with torch.no_grad():
            output=[]
            for i, (KEY_Y,VALUE_Y,QUERY_X, _ ) in enumerate(class_xyv_):
                KEY_Y = KEY_Y.to(self.device)
                VALUE_Y = VALUE_Y.to(self.device)
                QUERY_X = QUERY_X.to(self.device)

                tmp = self.model(QUERY_X.float(),VALUE_X.float(), KEY_Y.float(),VALUE_Y.float())[0]
                output.append(tmp)
            res = torch.cat(output, axis=0).to(self.device).detach()
        return res


class transformer_coords_steams(class_steams):
    def __init__(self,model,device):
        super(transformer_coords_steams, self).__init__(model,device)

    def single_train(self, data_loader):
        running_loss = 0.0
        self.model.train()
        for i, (KEY_Y,VALUE_Y,QUERY_X,VALUE_X) in enumerate(data_loader):

            KEY_Y = KEY_Y.to(self.device)
            VALUE_Y = VALUE_Y.to(self.device)
            QUERY_X = QUERY_X.to(self.device)
            VALUE_X = VALUE_X.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(QUERY_X.float(),KEY_Y.float(),VALUE_Y.float())
            loss = self.criterion(VALUE_X.float(),output[0])
            loss.backward()
            self.optimizer.step()
            self.scheduler_lr.step()

            if torch.isnan(loss) or loss == float('inf'):
                raise("Error infinite or NaN loss detected")
            running_loss += loss.item()
        avg_loss = running_loss / float(i)
        return avg_loss

    def loss(self,data_loader):
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for i, (KEY_Y,VALUE_Y,QUERY_X,VALUE_X) in enumerate(data_loader):

                KEY_Y = KEY_Y.to(self.device)
                VALUE_Y = VALUE_Y.to(self.device)
                QUERY_X = QUERY_X.to(self.device)
                VALUE_X = VALUE_X.to(self.device)

                output = self.model(QUERY_X.float(),KEY_Y.float(),VALUE_Y.float())[0]
                loss = self.criterion( VALUE_X.float(),output)
                running_loss += loss.item()
            avg_loss = running_loss / float(i)
        return avg_loss

    def evaluation_bytarget(self, data_loader, class_xyv_):
        self.model.eval()
        with torch.no_grad():
            tmp = pd.DataFrame()
            running_loss = np.zeros(2, dtype = int)
            for i, (KEY_Y,VALUE_Y,QUERY_X,VALUE_X) in enumerate(data_loader):
                KEY_Y = KEY_Y.to(self.device)
                VALUE_Y = VALUE_Y.to(self.device)
                QUERY_X = QUERY_X.to(self.device)
                VALUE_X = VALUE_X.to(self.device)

                output = self.model(QUERY_X.float(),KEY_Y.float(),VALUE_Y.float())[0]

                ##unscale
                #output_unscale = class_xyv_.unscale(output,"values")
                #target_unscale = class_xyv_.unscale(VALUE_X,"values")

                for k in range(output.shape[1]):
                    #loss = self.criterion(output_unscale[:,k], target_unscale[:,k])
                    loss = self.criterion(output[:,k], VALUE_X[:,k])
                    tmp.loc[i,k] = loss.item()
            avg_loss = pd.DataFrame({'crit':tmp.apply(lambda x: np.mean(x))})
        return avg_loss

    def predict(self, class_xyv_):
        self.model.eval()
        with torch.no_grad():
            output=[]
            for i, (KEY_Y,VALUE_Y,QUERY_X, _ ) in enumerate(class_xyv_):
                KEY_Y = KEY_Y.to(self.device)
                VALUE_Y = VALUE_Y.to(self.device)
                QUERY_X = QUERY_X.to(self.device)

                tmp = self.model(QUERY_X.float(),KEY_Y.float(),VALUE_Y.float())[0]
                output.append(tmp)
            res = torch.cat(output, axis=0).to(self.device).detach()
        return res
