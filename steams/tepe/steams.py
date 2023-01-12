import os
import torch

class class_steams():
    def __init__(self,model,device):
        '''
        class_steams enables to gather optimizer, scheduler, criterion in one object and simplify the design of function train, loss, evaluation and prediction.

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

    def load_model(self,path:str, name:str) -> None:
        model_path = os.path.join(path, name + "_model.pth")
        self.model.load_state_dict(torch.load(model_path))

class attention_steams(class_steams):
    def __init__(self,model,device):
        '''
        attention_steams provide function train, loss, evaluation and prediction for additive attention.

        Args:
            model:
            Model of class torch.nn.Module

            device:
            Determined with torch.device()
        '''

        super(attention_steams, self).__init__(model,device)

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
                raise("Error infinite or NaN loss detected")
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
            KEY_Y = KEY_Y.to(self.device)
            VALUE_Y = VALUE_Y.to(self.device)
            QUERY_X = QUERY_X.to(self.device)

            # input with dimension batch and on device
            KEY_Y = torch.reshape(KEY_Y,(1,KEY_Y.shape[0],KEY_Y.shape[1]))
            VALUE_Y = torch.reshape(VALUE_Y,(1,VALUE_Y.shape[0],VALUE_Y.shape[1]))
            QUERY_X = torch.reshape(QUERY_X,(1,QUERY_X.shape[0],QUERY_X.shape[1]))

            VALUE_X_pred = self.model(KEY_Y.float() ,VALUE_Y.float() ,QUERY_X.float() ).detach()

            VALUE_X_pred_unscaled = class_data.unscale(VALUE_X_pred,"VALUE_X").to(self.device)
            QUERY_X_unscaled = class_data.unscale(QUERY_X.detach(),"QUERY").to(self.device)

        return QUERY_X_unscaled, VALUE_X_pred_unscaled

class attention_ae_steams(class_steams):
    def __init__(self,model,device):
        super(attention_ae_steams, self).__init__(model,device)

    def single_train(self,data_loader):
        running_loss = 0.0
        self.model.train()
        for i, (KEY_Y,VALUE_Y,QUERY_X,VALUE_X) in enumerate(data_loader):

            KEY_Y = KEY_Y.to(self.device)
            VALUE_Y = VALUE_Y.to(self.device)
            QUERY_X = QUERY_X.to(self.device)
            VALUE_X = VALUE_X.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(KEY_Y.float() ,VALUE_Y.float() )

            loss_ = self.criterion(VALUE_Y.float(),output)
            loss_.backward()
            self.optimizer.step()
            #self.scheduler_lr.step()

            if torch.isnan(loss_) or loss_ == float('inf'):
                raise("Error infinite or NaN loss detected")
            running_loss += loss_.item()
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

                output = self.model(KEY_Y.float(),VALUE_Y.float())
                loss_ = self.criterion( VALUE_Y.float(),output)
                running_loss += loss_.item()
            avg_loss = running_loss / (float(i)+1.)
        return avg_loss

    def evaluation(self, data_loader, class_data):
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for i, (KEY_Y,VALUE_Y,QUERY_X,VALUE_X) in enumerate(data_loader):
                KEY_Y = KEY_Y.to(self.device)
                VALUE_Y = VALUE_Y.to(self.device)
                QUERY_X = QUERY_X.to(self.device)
                VALUE_X = VALUE_X.to(self.device)

                output = self.model(KEY_Y.float(),VALUE_Y.float())

                ##unscale
                output_unscale = class_data.unscale(output,"VALUE_Y").to(self.device)
                VALUE_Y_unscale = class_data.unscale(VALUE_Y,"VALUE_Y").to(self.device)

                loss_ = self.criterion( VALUE_Y_unscale.float(),output_unscale)
                running_loss += loss_.item()

            avg_loss = running_loss / (float(i)+1.)
        return avg_loss

    def predict(self, KEY_Y,VALUE_Y,class_data):
        self.model.eval()
        with torch.no_grad():
            KEY_Y = KEY_Y.to(self.device)
            VALUE_Y = VALUE_Y.to(self.device)

            # input with dimension batch and on device
            KEY_Y = torch.reshape(KEY_Y,(1,KEY_Y.shape[0],KEY_Y.shape[1]))
            VALUE_Y = torch.reshape(VALUE_Y,(1,VALUE_Y.shape[0],VALUE_Y.shape[1]))

            VALUE_Y_pred = self.model(KEY_Y.float() ,VALUE_Y.float()).detach()

            VALUE_Y_pred_unscaled = class_data.unscale(VALUE_Y_pred,"VALUE_Y").to(self.device)
            KEY_Y_unscaled = class_data.unscale(KEY_Y.detach(),"KEY").to(self.device)

        return KEY_Y_unscaled, VALUE_Y_pred_unscaled
