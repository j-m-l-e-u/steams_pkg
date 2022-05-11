import pandas as pd
import numpy as np
import torch
import random
import math
import os
from steams.utils.scale import param_scale
from steams.dictionnary.dictionnary_scaling import scaling_dict

class class_xyv_none():
    def __init__(self, params: dict,subset_indice=None ):

        path= params['path']
        tab_dir = os.path.join(path,'tab')
        coordinates=params['coordinates']
        values=params['values']
        self.nb_location = params['nb_location']
        self.history_length = params["features"]["history_length"]
        self.gap_length = params["target"]["gap_length"]
        self.horizon_length = params["target"]["horizon_length"]

        self.TO_SCALE = False

        tmp = pd.read_csv(os.path.join(tab_dir,'dataset.csv'))#.sort_values(by=date)

        ## subset indice
        if subset_indice is not None:
            self.indice = subset_indice
        else:
            self.indice = range(0,len(tmp))
        self.coordinates = coordinates
        self.df_coordinates = tmp.loc[:, self.coordinates]
        self.values = values
        self.df_values = tmp.loc[:, self.values]

        # min max scaling
        #self.scale_param_coordinates = param_scale(self.df_coordinates,'MinMaxScaler')
        self.scale_param_coordinates = param_scale(self.df_coordinates,'standardscaler')

        # Normilized, because kriging predict Gaussian values.
        self.scale_param_values = param_scale(self.df_values,'standardscaler')

        #
        self.len_xyt = len(self.df_values.index)

    def __getitem__(self, id):

        id_ = self.indice[id]

        # target:

        ## indice of the target:
        if ((self.history_length == 0) and (self.gap_length == 0) and (self.horizon_length == 0)):
            indice_target = id_
        else:
            range_min = id_ + self.nb_location * (self.history_length + self.gap_length)
            range_max = id_ + self.nb_location * (self.history_length + self.gap_length + self.horizon_length) #+1)
            range_max = min(range_max,self.len_xyt)
            indice_target = range(range_min,range_max,self.nb_location)

        ## coordinates (x,y,...)
        tmp = self.df_coordinates.iloc[indice_target]
        if self.TO_SCALE == True:
            #scaler = scaling_dict['MinMaxScaler']
            scaler = scaling_dict['StandardScaler']
            tmp = scaler(tmp, self.scale_param_coordinates, True)
        targ_coordinates_data = torch.from_numpy(tmp.to_numpy()).float()
        #targ_coordinates_data = torch.reshape(targ_coordinates_data, (1,3)).float()
        del(tmp)

        ## values
        tmp = self.df_values.iloc[indice_target]
        if self.TO_SCALE == True:
            scaler = scaling_dict['StandardScaler']
            tmp = scaler(tmp, self.scale_param_values, True)
            self.SCALED = True
        targ_values_data = torch.from_numpy(tmp.to_numpy()).float()
        #targ_values_data = torch.reshape(targ_values_data, (1,1)).float()
        del(tmp)


        # features
        range_min = self.nb_location * math.floor(id_/self.nb_location)
        range_max = self.nb_location * (math.floor(id_/self.nb_location) +1 + self.history_length )
        range_max = min(range_max,self.len_xyt)

        range_ = range(range_min,range_max)
        if ((self.history_length == 0) and (self.gap_length == 0) and (self.horizon_length == 0)):
            indice_features = random.sample([x for x in range_ if x != id_], min(len(range_),(self.nb_location-1)))
        else:
            indice_features = range(range_min,range_max)

        ## coordinates (x,y,...)
        tmp = self.df_coordinates.iloc[indice_features]
        if self.TO_SCALE == True:
            #scaler = scaling_dict['MinMaxScaler']
            scaler = scaling_dict['StandardScaler']
            tmp = scaler(tmp, self.scale_param_coordinates, True)
        feat_coordinates_data = torch.from_numpy(tmp.to_numpy()).float()
        del(tmp)

        ## values
        tmp = self.df_values.iloc[indice_features]
        if self.TO_SCALE == True:
            scaler = scaling_dict['StandardScaler']
            tmp = scaler(tmp, self.scale_param_values, True)
        feat_values_data = torch.from_numpy(tmp.to_numpy()).float()
        del(tmp)


        return feat_coordinates_data, feat_values_data, targ_coordinates_data, targ_values_data

    def get_scale_param_target(self):
        return self.scale_param_coordinates, self.scale_param_values

    def __len__(self) -> int:
        length = len(self.indice) - self.history_length - self.gap_length - self.horizon_length
        return length

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
            scaler = scaling_dict['StandardScaler']
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
