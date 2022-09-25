import os
import torch

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


#######
####### UNDER DEV, might change at any time
#######

class madsormer_steams(class_steams):
    def __init__(self,model,device):
        super(madsormer_steams, self).__init__(model,device)

    def single_train(self, data_loader):
        running_loss = 0.0
        self.model.train()
        for i, (KEY_Y, VALUE_Y_dec, QUERY_X, VALUE_X_enc, VALUE_X) in enumerate(data_loader):

            QUERY_X = QUERY_X.to(self.device)
            VALUE_X_enc = VALUE_X_enc.to(self.device)
            VALUE_X = VALUE_X.to(self.device)
            KEY_Y = KEY_Y.to(self.device)
            VALUE_Y_dec = VALUE_Y_dec.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(QUERY_X.float(),VALUE_X_enc.float(), KEY_Y.float(),VALUE_Y_dec.float())[0]
            loss_ = self.criterion(VALUE_X.float(),output)
            loss_.backward()
            self.optimizer.step()
            #self.scheduler_lr.step()

            if torch.isnan(loss_) or loss_ == float('inf'):
                raise("Error infinite or NaN loss detected")
            running_loss += loss_.item()
        avg_loss = running_loss / float(i)
        return avg_loss

    def loss(self,data_loader):
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for i, (KEY_Y, VALUE_Y_dec, QUERY_X, VALUE_X_enc, VALUE_X) in enumerate(data_loader):

                QUERY_X = QUERY_X.to(self.device)
                VALUE_X_enc = VALUE_X_enc.to(self.device)
                VALUE_X = VALUE_X.to(self.device)
                KEY_Y = KEY_Y.to(self.device)
                VALUE_Y_dec = VALUE_Y_dec.to(self.device)

                output = self.model(QUERY_X.float(),VALUE_X_enc.float(), KEY_Y.float(),VALUE_Y_dec.float())[0]
                loss_ = self.criterion( VALUE_X.float(),output)
                running_loss += loss_.item()
            avg_loss = running_loss / float(i)
        return avg_loss

    def evaluation(self, data_loader, class_data):
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for i, (KEY_Y, VALUE_Y_dec, QUERY_X, VALUE_X_enc, VALUE_X) in enumerate(data_loader):

                QUERY_X = QUERY_X.to(self.device)
                VALUE_X_enc = VALUE_X_enc.to(self.device)
                VALUE_X = VALUE_X.to(self.device)
                KEY_Y = KEY_Y.to(self.device)
                VALUE_Y_dec = VALUE_Y_dec.to(self.device)

                output = self.model(QUERY_X.float(),VALUE_X_enc.float(), KEY_Y.float(),VALUE_Y_dec.float())[0]

                #unscale
                output_unscale = class_data.unscale(output,"VALUE_X").to(self.device)
                VALUE_X_unscale = class_data.unscale(VALUE_X,"VALUE_X").to(self.device)

                loss_ = self.criterion( VALUE_X_unscale.float(),output_unscale)
                running_loss += loss_.item()

            avg_loss = running_loss / (float(i)+1.)
        return avg_loss

    def predict(self, KEY_Y,VALUE_Y_dec, QUERY_X, VALUE_X_enc,class_data):
        self.model.eval()
        with torch.no_grad():
            KEY_Y = KEY_Y.to(self.device)
            VALUE_Y_dec = VALUE_Y_dec.to(self.device)
            QUERY_X = QUERY_X.to(self.device)
            VALUE_X_enc = VALUE_X_enc.to(self.device)

            # input with dimension batch and on device
            KEY_Y = torch.reshape(KEY_Y,(1,KEY_Y.shape[0],KEY_Y.shape[1]))
            VALUE_Y_dec = torch.reshape(VALUE_Y_dec,(1,VALUE_Y_dec.shape[0],VALUE_Y_dec.shape[1]))
            QUERY_X = torch.reshape(QUERY_X,(1,QUERY_X.shape[0],QUERY_X.shape[1]))
            VALUE_X_enc = torch.reshape(VALUE_X_enc,(1,VALUE_X_enc.shape[0],VALUE_X_enc.shape[1]))

            VALUE_X_pred = self.model(QUERY_X.float(),VALUE_X_enc.float(), KEY_Y.float(),VALUE_Y_dec.float())[0].detach()

            VALUE_X_pred_unscaled = class_data.unscale(VALUE_X_pred,"VALUE_X").to(self.device)
            QUERY_X_unscaled = class_data.unscale(QUERY_X.detach(),"QUERY").to(self.device)

        return QUERY_X_unscaled, VALUE_X_pred_unscaled

#######
####### UNDER DEV, might change at any time
#######

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
            loss_ = self.criterion(VALUE_X.float(),output[0])
            loss_.backward()
            self.optimizer.step()
            self.scheduler_lr.step()

            if torch.isnan(loss) or loss == float('inf'):
                raise("Error infinite or NaN loss detected")
            running_loss += loss_.item()
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
                loss_ = self.criterion( VALUE_X.float(),output)
                running_loss += loss_.item()
            avg_loss = running_loss / float(i)
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

                output = self.model(QUERY_X.float(),KEY_Y.float(),VALUE_Y.float())[0]

                #unscale
                output_unscale = class_data.unscale(output,"VALUE_X").to(self.device)
                VALUE_X_unscale = class_data.unscale(VALUE_X,"VALUE_X").to(self.device)

                loss_ = self.criterion( VALUE_X_unscale.float(),output_unscale)
                running_loss += loss_.item()
            avg_loss = running_loss / (float(i)+1.)
        return avg_loss

    def predict(self, KEY_Y,VALUE_Y_dec, QUERY_X, VALUE_X_enc,class_data):
        self.model.eval()
        with torch.no_grad():
            KEY_Y = KEY_Y.to(self.device)
            VALUE_Y_dec = VALUE_Y_dec.to(self.device)
            QUERY_X = QUERY_X.to(self.device)
            VALUE_X_enc = VALUE_X_enc.to(self.device)

            # input with dimension batch and on device
            KEY_Y = torch.reshape(KEY_Y,(1,KEY_Y.shape[0],KEY_Y.shape[1]))
            VALUE_Y_dec = torch.reshape(VALUE_Y_dec,(1,VALUE_Y_dec.shape[0],VALUE_Y_dec.shape[1]))
            QUERY_X = torch.reshape(QUERY_X,(1,QUERY_X.shape[0],QUERY_X.shape[1]))
            VALUE_X_enc = torch.reshape(VALUE_X_enc,(1,VALUE_X_enc.shape[0],VALUE_X_enc.shape[1]))

            VALUE_X_pred = self.model(QUERY_X.float(),VALUE_X_enc.float(), KEY_Y.float(),VALUE_Y_dec.float())[0].detach()

            VALUE_X_pred_unscaled = class_data.unscale(VALUE_X_pred,"VALUE_X").to(self.device)
            QUERY_X_unscaled = class_data.unscale(QUERY_X.detach(),"QUERY").to(self.device)

        return QUERY_X_unscaled, VALUE_X_pred_unscaled
