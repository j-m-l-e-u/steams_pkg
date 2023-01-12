import pandas as pd
import numpy as np
import torch
import random
import math
import os
import numpy as np
from steams.utils.scale import param_scale,standard_scaler

class KVyQVx():
    def __init__(self, params: dict,subset_indice=None ):
        '''
        KVyQVx is a class used as data sampler.
        It samples keys, values related to these keys, queries, and values related to these queries from csv files.
        Input csv files describes sparse space-time values with a constant number of point for each time step (E.g. 10 stations for 5 time steps).

        Args:
            params:
            Dictionary providing information about network X and network Y.
            Information related to Y are: the path of file 'dataset.csv', the name of the columns used as Key, the number of locations in the network, the history length, the number pair {K,Vy} to sample.
            Information related to X are: the path of file 'dataset.csv', the name of the columns used as Query, the number of locations in the network, the gap length, the horizon length, the number pair {Q,Vx} to sample.

            subset_indice:
            A sequence of integer that describes the sub-sample of the csv file to use.
        '''

        self.TO_SCALE = False

        ################
        # FEATURES: Y  #
        ################

        path_Y= params['Y']['path']
        tab_Y_dir = os.path.join(path_Y,'tab')
        self.tmp_Y = pd.read_csv(os.path.join(tab_Y_dir,'dataset.csv'))

        self.KEY = params['Y']['KEY']
        self.df_KEY = self.tmp_Y.loc[:, self.KEY]

        self.VALUE_Y = params['Y']['VALUE']
        self.df_VALUE_Y = self.tmp_Y.loc[:, self.VALUE_Y]

        self.nb_location_Y = params['Y']['nb_location']
        self.history_length_Y = params['Y']['history_length']
        self.nb_sampling_Y = params['Y']['nb_sampling']

        # Scaling
        self.scale_param_KEY = param_scale(self.df_KEY,'standardscaler')#'standardscaler')
        self.scale_param_VALUE_Y = param_scale(self.df_VALUE_Y,'standardscaler')

        self.indice_Y = range(0,len(self.tmp_Y))

        # nb length
        self.len_VALUE_Y = len(self.df_VALUE_Y.index)

        ################
        # TARGET: X    #
        ################

        path_X= params['X']['path']
        tab_X_dir = os.path.join(path_X,'tab')
        self.tmp_X = pd.read_csv(os.path.join(tab_X_dir,'dataset.csv'))

        self.QUERY = params['X']['QUERY']
        self.df_QUERY = self.tmp_X.loc[:, self.QUERY]

        self.VALUE_X = params['X']['VALUE']
        self.df_VALUE_X = self.tmp_X.loc[:, self.VALUE_X]

        self.nb_location_X = params['X']['nb_location']
        self.gap_length_X = params['X']['gap_length']
        self.horizon_length_X = params['X']['horizon_length']
        self.nb_sampling_X = params['X']['nb_sampling']

        # Scaling
        self.scale_param_QUERY = param_scale(self.df_QUERY,'standardscaler')#'standardscaler')
        self.scale_param_VALUE_X = param_scale(self.df_VALUE_X,'standardscaler')

        ## subset indice, rem: X decide for Y
        if subset_indice is not None:
            self.indice_X = subset_indice
        else:
            self.indice_X = range(0,len(self.tmp_X))

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
        self.indice_X_ = random.sample([x for x in range_ ], min(len(range_),(self.nb_sampling_X)))

        ## QUERY
        tmp = self.df_QUERY.iloc[self.indice_X_]
        if self.TO_SCALE == True:
            scaler = standard_scaler
            tmp = scaler(tmp, self.scale_param_QUERY, True)
        QUERY_data = torch.from_numpy(tmp.to_numpy()).float()
        del(tmp)

        ## VALUE
        tmp = self.df_VALUE_X.loc[self.indice_X_,self.VALUE_X]
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
        indice_Y_ = random.sample([x for x in range_ ], min(len(range_),(self.nb_sampling_Y)))

        ## KEY
        tmp = self.df_KEY.iloc[indice_Y_]
        if self.TO_SCALE == True:
            scaler = standard_scaler
            tmp = scaler(tmp, self.scale_param_KEY, True)
        KEY_data = torch.from_numpy(tmp.to_numpy()).float()
        del(tmp)

        ## VALUE
        tmp = self.df_VALUE_Y.loc[indice_Y_,self.VALUE_Y]
        if self.TO_SCALE == True:
            scaler = standard_scaler
            tmp = scaler(tmp, self.scale_param_VALUE_Y, True)
        VALUE_Y_data = torch.from_numpy(tmp.to_numpy()).float()
        del(tmp)

        return KEY_data, VALUE_Y_data, QUERY_data, VALUE_X_data

    def get_rand_input(self):
        '''
        Function used when converting the model into ONNX format
        '''

        #############
        # target: X #
        #############
        id_X = self.indice_X[0]

        ## indice of the target:
        range_min = self.nb_location_X * (math.floor(id_X/self.nb_location_X) + self.history_length_Y + self.gap_length_X)
        range_max = self.nb_location_X * (math.floor(id_X/self.nb_location_X) + 1 + self.history_length_Y + self.gap_length_X + self.horizon_length_X)
        range_max = min(range_max,self.len_VALUE_X)
        range_ = range(range_min,range_max)
        indice_X_ = random.sample([x for x in range_ ], min(len(range_),(self.nb_sampling_X)))

        ## QUERY
        tmp = self.df_QUERY.iloc[indice_X_]
        QUERY_data = torch.from_numpy(tmp.to_numpy()).float()
        del(tmp)

        ## VALUE_X
        tmp = self.df_VALUE_X.loc[indice_X_,self.VALUE_X]
        VALUE_X_data = torch.from_numpy(tmp.to_numpy()).float()
        del(tmp)

        ############
        # features #
        ############
        id_Y = math.floor(id_X * self.len_VALUE_Y/self.len_VALUE_X )

        range_min = self.nb_location_Y * math.floor(id_Y/self.nb_location_Y)
        range_max = self.nb_location_Y * (math.floor(id_Y/self.nb_location_Y) +1 + self.history_length_Y )
        range_max = min(range_max,self.len_VALUE_Y)
        range_ = range(range_min,range_max)
        indice_Y_ = random.sample([x for x in range_ ], min(len(range_),(self.nb_sampling_Y)))

        ## coordinates (x,y,...)
        tmp = self.df_KEY.iloc[indice_Y_]
        KEY_data = torch.from_numpy(tmp.to_numpy()).float()
        del(tmp)

        ## values
        tmp = self.df_VALUE_Y.loc[indice_Y_,self.VALUE_Y]
        VALUE_Y_data = torch.from_numpy(tmp.to_numpy()).float()
        del(tmp)

        return (KEY_data, VALUE_Y_data, QUERY_data, VALUE_X_data)

    def __len__(self) -> int:
        return len(self.indice_X) - self.nb_location_X*(self.history_length_Y + self.gap_length_X + self.horizon_length_X)

    def scale(self,SCALE:bool):
        '''
        Determines whether to scale or not the sampled data. By default the data sample does not scale anything.
        Args:
            SCALE: Boolean chosen either 'True' to scale or 'False' to not scale.
        '''
        self.TO_SCALE = SCALE

    def unscale(self, newdata=None, datatype = None):
        '''
        Unscales a dataset using sclae parameters
        Args:
            newdata: the dataset to unscale
            datatype: type of the dataset to uncsale; either 'KEY','VALUE_Y','QUERY' or 'VALUE_X'.
        '''
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
        elif datatype == 'VALUE_Y':
            scaler = standard_scaler
            tmp = scaler(tmp, self.scale_param_VALUE_Y, False)
            if isinstance(newdata, torch.Tensor):
                res = torch.from_numpy(tmp)
            elif isinstance(newdata, pd.Series) or isinstance(newdata, pd.DataFrame):
                res = pd.DataFrame(tmp, index=newdata.index, columns=newdata.columns)
            elif isinstance(newdata, np.ndarray):
                res = tmp
            else:
                print('instance of tmp not known')
        elif datatype == 'QUERY' :
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
            print('datatype either KEY, VALUE_Y, QUERY or VALUE_X')
        return res
