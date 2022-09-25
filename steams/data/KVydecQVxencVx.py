import pandas as pd
import numpy as np
import torch
import random
import math
import os
import numpy as np
from steams.utils.scale import param_scale,standard_scaler

class KVydecQVxencVx():
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

        self.VALUE_Y_dec = params['Y']['VALUE_dec']
        self.df_VALUE_Y_dec = tmp_Y.loc[:, self.VALUE_Y_dec]

        self.nb_location_Y = params['Y']['nb_location']
        self.history_length_Y = params['Y']['history_length']
        self.nb_sampling_Y = params['Y']['nb_sampling']

        # Scaling
        self.scale_param_KEY = param_scale(self.df_KEY,'standardscaler')
        self.scale_param_VALUE_Y_dec = param_scale(self.df_VALUE_Y_dec,'standardscaler')

        self.indice_Y = range(0,len(tmp_Y))

        # nb length
        self.len_VALUE_Y = len(self.df_VALUE_Y_dec.index)

        ################
        # TARGET: X    #
        ################

        path_X= params['X']['path']
        tab_X_dir = os.path.join(path_X,'tab')
        tmp_X = pd.read_csv(os.path.join(tab_X_dir,'dataset.csv'))

        self.QUERY = params['X']['QUERY']
        self.df_QUERY = tmp_X.loc[:, self.QUERY]

        self.VALUE_X_enc = params['X']['VALUE_enc']
        self.df_VALUE_X_enc = tmp_X.loc[:, self.VALUE_X_enc]

        self.VALUE_X = params['X']['VALUE']
        self.df_VALUE_X = tmp_X.loc[:, self.VALUE_X]

        self.nb_location_X = params['X']['nb_location']
        self.gap_length_X = params['X']['gap_length']
        self.horizon_length_X = params['X']['horizon_length']
        self.nb_sampling_X = params['X']['nb_sampling']

        # Scaling
        self.scale_param_QUERY = param_scale(self.df_QUERY,'standardscaler')
        self.scale_param_VALUE_X_enc = param_scale(self.df_VALUE_X_enc,'standardscaler')
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
        range_min = self.nb_location_X * (math.floor(id_X/self.nb_location_X) + self.history_length_Y + self.gap_length_X)
        range_max = self.nb_location_X * (math.floor(id_X/self.nb_location_X) + 1 + self.history_length_Y + self.gap_length_X + self.horizon_length_X)
        range_max = min(range_max,self.len_VALUE_X)
        range_ = range(range_min,range_max)
        indice_X = random.sample([x for x in range_ ], min(len(range_),(self.nb_sampling_X)))

        ## QUERY
        tmp = self.df_QUERY.iloc[indice_X]
        if self.TO_SCALE == True:
            scaler = standard_scaler
            tmp = scaler(tmp, self.scale_param_QUERY, True)
        QUERY_data = torch.from_numpy(tmp.to_numpy()).float()
        del(tmp)

        ## VALUE_enc_X
        tmp = self.df_VALUE_X_enc.loc[indice_X,self.VALUE_X_enc]
        if self.TO_SCALE == True:
            scaler = standard_scaler
            tmp = scaler(tmp, self.scale_param_VALUE_X_enc, True)
            self.SCALED = True
        VALUE_X_enc_data = torch.from_numpy(tmp.to_numpy()).float()
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

        range_min = self.nb_location_Y * math.floor(id_Y/self.nb_location_Y)
        range_max = self.nb_location_Y * (math.floor(id_Y/self.nb_location_Y) +1 + self.history_length_Y )
        range_max = min(range_max,self.len_VALUE_Y)
        range_ = range(range_min,range_max)
        indice_Y = random.sample([x for x in range_ ], min(len(range_),(self.nb_sampling_Y)))

        ## coordinates (x,y,...)
        tmp = self.df_KEY.iloc[indice_Y]
        if self.TO_SCALE == True:
            scaler = standard_scaler
            tmp = scaler(tmp, self.scale_param_KEY, True)
        KEY_data = torch.from_numpy(tmp.to_numpy()).float()
        del(tmp)

        ## values
        tmp = self.df_VALUE_Y_dec.loc[indice_Y,self.VALUE_Y_dec]
        if self.TO_SCALE == True:
            scaler = standard_scaler
            tmp = scaler(tmp, self.scale_param_VALUE_Y_dec, True)
        VALUE_Y_dec_data = torch.from_numpy(tmp.to_numpy()).float()
        del(tmp)

        return KEY_data, VALUE_Y_dec_data, QUERY_data, VALUE_X_enc_data, VALUE_X_data

    def get_rand_input(self):

        #############
        # target: X #
        #############
        id_X = self.indice_X[0]

        ## indice of the target:
        range_min = self.nb_location_X * (math.floor(id_X/self.nb_location_X) + self.history_length_Y + self.gap_length_X)
        range_max = self.nb_location_X * (math.floor(id_X/self.nb_location_X) + 1 + self.history_length_Y + self.gap_length_X + self.horizon_length_X)
        range_max = min(range_max,self.len_VALUE_X)
        range_ = range(range_min,range_max)
        indice_X = random.sample([x for x in range_ ], min(len(range_),(self.nb_sampling_X)))

        ## QUERY
        tmp = self.df_QUERY.iloc[indice_X]
        QUERY_data = torch.from_numpy(tmp.to_numpy()).float()
        del(tmp)

        ## VALUE_enc_X
        tmp = self.df_VALUE_X_enc.loc[indice_X,self.VALUE_X_enc]
        VALUE_X_enc_data = torch.from_numpy(tmp.to_numpy()).float()
        del(tmp)

        ## VALUE_X
        tmp = self.df_VALUE_X.loc[indice_X,self.VALUE_X]
        VALUE_X_data = torch.from_numpy(tmp.to_numpy()).float()
        del(tmp)

        ##############
        # features:Y #
        ##############

        id_Y = math.floor(id_X * self.len_VALUE_Y/self.len_VALUE_X )

        range_min = self.nb_location_Y * math.floor(id_Y/self.nb_location_Y)
        range_max = self.nb_location_Y * (math.floor(id_Y/self.nb_location_Y) +1 + self.history_length_Y )
        range_max = min(range_max,self.len_VALUE_Y)
        range_ = range(range_min,range_max)
        indice_Y = random.sample([x for x in range_ ], min(len(range_),(self.nb_sampling_Y)))

        ## coordinates (x,y,...)
        tmp = self.df_KEY.iloc[indice_Y]
        KEY_data = torch.from_numpy(tmp.to_numpy()).float()
        del(tmp)

        ## values
        tmp = self.df_VALUE_Y_dec.loc[indice_Y,self.VALUE_Y_dec]
        VALUE_Y_dec_data = torch.from_numpy(tmp.to_numpy()).float()
        del(tmp)

        return (KEY_data, VALUE_Y_dec_data, QUERY_data, VALUE_X_enc_data, VALUE_X_data)

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

        if datatype == 'KEY' :
            scaler = standard_scaler
            tmp = scaler(tmp, self.scale_param_KEY, False)
            if isinstance(newdata, torch.Tensor):
                res = torch.from_numpy(tmp)
            elif isinstance(newdata, pd.Series) or isinstance(newdata, pd.DataFrame):
                res = pd.DataFrame(tmp, index=newdata.index, columns=newdata.columns)
            elif isinstance(newdata, np.ndarray):
                res = tmp
            else:
                print('instance of tmp not known')
        elif datatype == 'VALUE_Y_dec':
            scaler = standard_scaler
            tmp = scaler(tmp, self.scale_param_VALUE_Y_dec, False)
            if isinstance(newdata, torch.Tensor):
                res = torch.from_numpy(tmp)
            elif isinstance(newdata, pd.Series) or isinstance(newdata, pd.DataFrame):
                res = pd.DataFrame(tmp, index=newdata.index, columns=newdata.columns)
            elif isinstance(newdata, np.ndarray):
                res = tmp
            else:
                print('instance of tmp not known')
        if datatype == 'QUERY' :
            scaler = standard_scaler
            tmp = scaler(tmp, self.scale_param_QUERY, False)
            if isinstance(newdata, torch.Tensor):
                res = torch.from_numpy(tmp)
            elif isinstance(newdata, pd.Series) or isinstance(newdata, pd.DataFrame):
                res = pd.DataFrame(tmp, index=newdata.index, columns=newdata.columns)
            elif isinstance(newdata, np.ndarray):
                res = tmp
            else:
                print('instance of tmp not known')
        elif datatype == 'VALUE_X_enc':
            scaler = standard_scaler
            tmp = scaler(tmp, self.scale_param_VALUE_X_enc, False)
            if isinstance(newdata, torch.Tensor):
                res = torch.from_numpy(tmp)
            elif isinstance(newdata, pd.Series) or isinstance(newdata, pd.DataFrame):
                res = pd.DataFrame(tmp, index=newdata.index, columns=newdata.columns)
            elif isinstance(newdata, np.ndarray):
                res = tmp
            else:
                print('instance of tmp not known')
        elif datatype == 'VALUE_X':
            scaler = standard_scaler
            tmp = scaler(tmp, self.scale_param_VALUE_X, False)
            if isinstance(newdata, torch.Tensor):
                res = torch.from_numpy(tmp)
            elif isinstance(newdata, pd.Series) or isinstance(newdata, pd.DataFrame):
                res = pd.DataFrame(tmp, index=newdata.index, columns=newdata.columns)
            elif isinstance(newdata, np.ndarray):
                res = tmp
            else:
                print('instance of tmp not known')
        else:
            print('datatype either KEY, VALUE_Y_dec, QUERY, VALUE_X_enc, VALUE_X')
        return res
