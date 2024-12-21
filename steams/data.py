import pandas as pd
# import numpy as np
import torch
import random
import math
import os

from torch.utils.data import Dataset



# import polars as pl

#     def batch_generator(self, batch_size):
#         """
#         Yields batches of sampled data.
#         """
#         n_batches = len(self) // batch_size
#         for i in range(n_batches):
#             batch_indices = range(i * batch_size, (i + 1) * batch_size)
#             batch_data = [self[idx] for idx in batch_indices]

#             # Combine batch tensors
#             KEY, VALUE_Y, QUERY, VALUE_X = zip(*batch_data)
#             yield (
#                 torch.stack(KEY),
#                 torch.stack(VALUE_Y),
#                 torch.stack(QUERY),
#                 torch.stack(VALUE_X),
#             )


class KVyQVx(Dataset):
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

        ################
        # FEATURES: Y  #
        ################

        path_Y= params['Y']['path']
        tab_Y_dir = os.path.join(path_Y,'tab')
        self.tmp_Y = pd.read_csv(os.path.join(tab_Y_dir,'dataset.csv'))

        self.KEY = params['Y']['KEY']
        self.df_KEY = self.tmp_Y.loc[:, self.KEY]
        self.tensor_KEY = torch.tensor(self.df_KEY.values, dtype=torch.float32)

        self.VALUE_Y = params['Y']['VALUE']
        self.df_VALUE_Y = self.tmp_Y.loc[:, self.VALUE_Y]
        self.tensor_VALUE_Y = torch.tensor(self.df_VALUE_Y.values, dtype=torch.float32)

        self.nb_location_Y = params['Y']['nb_location']
        self.history_length_Y = params['Y']['history_length']
        self.nb_sampling_Y = params['Y']['nb_sampling']

        # Scaling
        self.mean_KEY = self.tensor_KEY.mean(dim=0, keepdim=True)
        self.std_KEY = self.tensor_KEY.std(dim=0, keepdim=True)
        self.std_KEY[self.std_KEY == 0] = 1

        self.mean_VALUE_Y = self.tensor_VALUE_Y.mean(dim=0, keepdim=True)
        self.std_VALUE_Y = self.tensor_VALUE_Y.std(dim=0, keepdim=True)
        self.std_VALUE_Y[self.std_VALUE_Y == 0] = 1

    
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
        self.tensor_QUERY = torch.tensor(self.df_QUERY.values, dtype=torch.float32)

        self.VALUE_X = params['X']['VALUE']
        self.df_VALUE_X = self.tmp_X.loc[:, self.VALUE_X]
        self.tensor_VALUE_X = torch.tensor(self.df_VALUE_X.values, dtype=torch.float32)

        self.nb_location_X = params['X']['nb_location']
        self.gap_length_X = params['X']['gap_length']
        self.horizon_length_X = params['X']['horizon_length']
        self.nb_sampling_X = params['X']['nb_sampling']

        # Scaling
        self.mean_QUERY = self.tensor_QUERY.mean(dim=0, keepdim=True)
        self.std_QUERY = self.tensor_QUERY.std(dim=0, keepdim=True)
        self.std_QUERY[self.std_QUERY == 0] = 1

        self.mean_VALUE_X = self.tensor_VALUE_X.mean(dim=0, keepdim=True)
        self.std_VALUE_X = self.tensor_VALUE_X.std(dim=0, keepdim=True)
        self.std_VALUE_X[self.std_VALUE_X == 0] = 1

        ## subset indice, rem: X decide for Y
        if subset_indice is not None:
            self.indice_X = subset_indice
        else:
            self.indice_X = range(0,len(self.tmp_X))

        # nb length
        self.len_VALUE_X = len(self.df_VALUE_X.index)

    def scale(self, apply_scaling: bool):
        """
        Scales VALUE_X and VALUE_Y columns if apply_scaling is True.
        """
        if apply_scaling:

            self.tensor_QUERY = (self.tensor_QUERY - self.mean_QUERY) / self.std_QUERY
            self.tensor_VALUE_X = (self.tensor_VALUE_X - self.mean_VALUE_X) / self.std_VALUE_X
            self.tensor_KEY = (self.tensor_KEY - self.mean_KEY) / self.std_KEY
            self.tensor_VALUE_Y = (self.tensor_VALUE_Y - self.mean_VALUE_Y) / self.std_VALUE_Y
            
        else:
            self.tensor_QUERY = (self.tensor_QUERY *self.std_QUERY + self.mean_QUERY) 
            self.tensor_VALUE_X = (self.tensor_VALUE_X * self.std_VALUE_X + self.mean_VALUE_X)
            self.tensor_KEY = (self.tensor_KEY * self.std_KEY + self.mean_KEY)
            self.tensor_VALUE_Y = (self.tensor_VALUE_Y * self.std_VALUE_Y + self.mean_VALUE_Y)

    def __getitem__(self, id):
        id_X = self.indice_X[id]

        # Query and VALUE_X Sampling
        start_X = self.nb_location_X * (id_X // self.nb_location_X + self.history_length_Y + self.gap_length_X)
        end_X = start_X + self.nb_location_X * (self.horizon_length_X + 1)
        sampled_indices_X = torch.randint(start_X, min(end_X, self.len_VALUE_X), (self.nb_sampling_X,))

        QUERY_data = self.tensor_QUERY[sampled_indices_X]
        VALUE_X_data = self.tensor_VALUE_X[sampled_indices_X]
        
        # Key and VALUE_Y Sampling
        id_Y = int(id_X * self.len_VALUE_Y / self.len_VALUE_X)
        start_Y = self.nb_location_Y * (id_Y // self.nb_location_Y)
        end_Y = start_Y + self.nb_location_Y * (self.history_length_Y + 1)
        sampled_indices_Y = torch.randint(start_Y, min(end_Y, self.len_VALUE_Y), (self.nb_sampling_Y,))

        KEY_data = self.tensor_KEY[sampled_indices_Y]
        VALUE_Y_data = self.tensor_VALUE_Y[sampled_indices_Y]
        
        return KEY_data, VALUE_Y_data, QUERY_data, VALUE_X_data

    def __len__(self) -> int:
        return len(self.indice_X) - self.nb_location_X*(self.history_length_Y + self.gap_length_X + self.horizon_length_X)

    
    def unscale(self, newdata=None, datatype = None):
        '''
        Unscales a dataset using scale parameters
        Args:
            newdata: the dataset to unscale
            datatype: type of the dataset to uncsale; either 'KEY','VALUE_Y','QUERY' or 'VALUE_X'.
        '''
        
        if datatype == 'KEY' :
            res = (newdata * self.std_KEY + self.mean_KEY)
        elif datatype == 'VALUE_Y':
            res = (newdata * self.std_VALUE_Y + self.mean_VALUE_Y)
        elif datatype == 'QUERY' :
            res = (newdata * self.std_QUERY + self.mean_QUERY)
        elif datatype == 'VALUE_X':
            res = (newdata * self.std_VALUE_X + self.mean_VALUE_X)
        else:
            print('datatype either KEY, VALUE_Y, QUERY or VALUE_X')
        return res
