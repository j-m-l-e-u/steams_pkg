import pandas as pd
import numpy as np
import torch
import random
import math
import os
import numpy as np
from steams.utils.scale import param_scale,standard_scaler

class irKVyQVx():
    def __init__(self, params: dict,subset_indice=None ):

        self.TO_SCALE = False

        ################
        # FEATURES: Y  #
        ################

        path_Y= params['Y']['path']
        tab_Y_dir = os.path.join(path_Y,'tab')
        tmp_Y = pd.read_csv(os.path.join(tab_Y_dir,'dataset.csv'))

        self.KEY = params['Y']['KEY']
        self.df_KEY = tmp_Y.loc[:, self.KEY]

        self.VALUE_Y = params['Y']['VALUE']
        self.df_VALUE_Y = tmp_Y.loc[:, self.VALUE_Y]

        self.nb_sampling_Y = params['Y']['nb_sampling']

        # Scaling
        self.scale_param_KEY = param_scale(self.df_KEY,'standardscaler')#'standardscaler')
        self.scale_param_VALUE_Y = param_scale(self.df_VALUE_Y,'standardscaler')

        self.indice_Y = range(0,len(tmp_Y))

        # nb length
        self.len_VALUE_Y = len(self.df_VALUE_Y.index)

        ################
        # TARGET: X    #
        ################

        path_X= params['X']['path']
        tab_X_dir = os.path.join(path_X,'tab')
        tmp_X = pd.read_csv(os.path.join(tab_X_dir,'dataset.csv'))

        self.QUERY = params['X']['QUERY']
        self.df_QUERY = tmp_X.loc[:, self.QUERY]

        self.VALUE_X = params['X']['VALUE']
        self.df_VALUE_X = tmp_X.loc[:, self.VALUE_X]

        self.nb_sampling_X = params['X']['nb_sampling']

        # Scaling
        self.scale_param_QUERY = param_scale(self.df_QUERY,'standardscaler')#'standardscaler')
        self.scale_param_VALUE_X = param_scale(self.df_VALUE_X,'standardscaler')

        ## subset indice, rem: X decide for Y
        if subset_indice is not None:
            self.indice_X = subset_indice
        else:
            self.indice_X = range(0,len(tmp_X))

        # nb length
        self.len_VALUE_X = len(self.df_VALUE_X.index)

    def __getitem__(self, id):

        #############
        # target: X #
        #############
        id_X = self.indice_X[id]

        ## indice of the target:
        range_ = range(id_X,self.len_VALUE_X)
        indice_X = random.sample([x for x in range_ ], min(len(range_),(self.nb_sampling_X)))

        ## QUERY
        tmp = self.df_QUERY.iloc[indice_X]
        if self.TO_SCALE == True:
            scaler = standard_scaler
            tmp = scaler(tmp, self.scale_param_QUERY, True)
        QUERY_data = torch.from_numpy(tmp.to_numpy()).float()
        del(tmp)

        ## VALUE_X
        tmp = self.df_VALUE_X.loc[indice_X,self.VALUE_X]
        if self.TO_SCALE == True:
            scaler = standard_scaler
            tmp = scaler(tmp, self.scale_param_VALUE_X, True)
            self.SCALED = True
        VALUE_X_data = torch.from_numpy(tmp.to_numpy()).float()
        del(tmp)

        ##############
        # features:Y #
        ##############

        id_Y = math.floor(id_X * self.len_VALUE_Y/self.len_VALUE_X )

        range_ = range(id_Y,self.len_VALUE_Y)
        indice_Y = random.sample([x for x in range_ ], min(len(range_),(self.nb_sampling_Y)))

        ## coordinates (x,y,...)
        tmp = self.df_KEY.iloc[indice_Y]
        if self.TO_SCALE == True:
            scaler = standard_scaler
            tmp = scaler(tmp, self.scale_param_KEY, True)
        KEY_data = torch.from_numpy(tmp.to_numpy()).float()
        del(tmp)

        ## values
        tmp = self.df_VALUE_Y.loc[indice_Y,self.VALUE_Y]
        if self.TO_SCALE == True:
            scaler = standard_scaler
            tmp = scaler(tmp, self.scale_param_VALUE_Y, True)
        VALUE_Y_data = torch.from_numpy(tmp.to_numpy()).float()
        del(tmp)

        return KEY_data, VALUE_Y_data, QUERY_data, VALUE_X_data

    def get_rand_input(self):

        #############
        # target: X #
        #############
        id_X = self.indice_X[0]

        ## indice of the target:
        range_ = range(id_X,self.len_VALUE_X)
        indice_X = random.sample([x for x in range_ ], min(len(range_),(self.nb_sampling_X)))

        ## QUERY
        tmp = self.df_QUERY.iloc[indice_X]
        QUERY_data = torch.from_numpy(tmp.to_numpy()).float()
        del(tmp)

        ## VALUE_X
        tmp = self.df_VALUE_X.loc[indice_X,self.VALUE_X]
        VALUE_X_data = torch.from_numpy(tmp.to_numpy()).float()
        del(tmp)

        ##############
        # features:Y #
        ##############

        id_Y = math.floor(id_X * self.len_VALUE_Y/self.len_VALUE_X )

        range_ = range(id_Y,self.len_VALUE_Y)
        indice_Y = random.sample([x for x in range_ ], min(len(range_),(self.nb_sampling_Y)))

        ## coordinates (x,y,...)
        tmp = self.df_KEY.iloc[indice_Y]
        KEY_data = torch.from_numpy(tmp.to_numpy()).float()
        del(tmp)

        ## values
        tmp = self.df_VALUE_Y.loc[indice_Y,self.VALUE_Y]
        VALUE_Y_data = torch.from_numpy(tmp.to_numpy()).float()
        del(tmp)

        return (KEY_data, VALUE_Y_data, QUERY_data, VALUE_X_data)

    def get_scale_param_target(self):
        return self.scale_param_coordinates, self.scale_param_values

    def __len__(self) -> int:
        return len(self.indice_X) - self.nb_location_X*(self.history_length_Y + self.gap_length_X + self.horizon_length_X)

    def scale(self,SCALE:bool):
        self.TO_SCALE = SCALE

    def unscale(self, newdata=None, datatype = None):
        if isinstance(newdata, torch.Tensor):
            tmp = newdata.cpu().numpy()
        elif isinstance(newdata, pd.Series) or isinstance(newdata, pd.DataFrame):
            tmp = np.asarray(newdata)
        elif isinstance(newdata, np.ndarray):
            tmp = newdata
        else:
            print('instance of newdata not known')
        if datatype == 'values' :
            scaler = standard_scaler
            res = scaler(tmp, self.scale_param_values, False)
            if isinstance(newdata, torch.Tensor):
                res = torch.from_numpy(tmp)
            elif isinstance(newdata, pd.Series) or isinstance(newdata, pd.DataFrame):
                res = pd.DataFrame(tmp, index=newdata.index, columns=newdata.columns)
            elif isinstance(newdata, np.ndarray):
                res = tmp
            else:
                print('instance of tmp not known')
        elif datatype == 'coordinates':
            scaler = scaling_dict['MinMaxScaler']
            res = scaler(tmp, self.scale_param_features, False)
            if isinstance(newdata, torch.Tensor):
                res = torch.from_numpy(tmp)
            elif isinstance(newdata, pd.Series) or isinstance(newdata, pd.DataFrame):
                res = pd.DataFrame(tmp, index=newdata.index, columns=newdata.columns)
            elif isinstance(newdata, np.ndarray):
                res = tmp
            else:
                print('instance of tmp not known')
        else:
            print('datatype either values or coordinates')
        return res
