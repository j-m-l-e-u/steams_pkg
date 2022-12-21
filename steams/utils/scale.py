import pandas as pd
import numpy as np

# why writting this functions:
# --> self.scaling_dict[scaling] is defined as a global variable;consequence: scaling features is replaced by scaling targ
#self.scalefeat = self.scaling_dict[scaling]
#self.scalefeat.fit(self.df_features)
#tmp_df = self.scalefeat.transform(self.df_features)
#self.df_features = pd.DataFrame(tmp_df, index=self.df_features.index, columns=self.df_features.columns)
#del tmp_df
#self.scaletarg = self.scaling_dict[scaling]
#self.scaletarg.fit(self.df_target)
#tmp_df = self.scaletarg.transform(self.df_target)
#self.df_target = pd.DataFrame(tmp_df, index=self.df_target.index, columns=self.df_target.columns)
#del tmp_df

def standard_scaler(data, param, Scale):
    if isinstance(data, pd.DataFrame):
        res = standard_scaler_pd(data, param['mean'], param['std'], Scale)
    elif isinstance(data, np.ndarray):
        res = standard_scaler_ndarray(data, param['mean'], param['std'], Scale)
    return res

def minmax_scaler(data, param, Scale):
    if isinstance(data, pd.DataFrame):
        res = minmax_scaler_pd(data, param['min'], param['max'], Scale)
    elif isinstance(data, np.ndarray):
        res = minmax_scaler_ndarray(data, param['min'], param['max'], Scale)
    return res


def standard_scaler_pd(data, mean_data, std_data, Scale):
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    tmp = data.copy()
    if Scale is True:
        for i in data:
            #tmp = data.apply(lambda x: (x - mean_data[i])/ std_data[i])
            #data = tmp
            tmp[i] = (data[i]-mean_data[i][0])/ std_data[i][0]
    if Scale is False:
        for i in data:
            #tmp = data.apply(lambda x: (x *std_data[i] + mean_data[i]))
            #data = tmp
            tmp[i] = data[i]*std_data[i][0] + mean_data[i][0]
    return tmp

def standard_scaler_ndarray(data, mean_data, std_data, Scale):
    if Scale is True:
        if len(data.shape)==3:
            for i in range(mean_data.shape[1]):
                data[:,:,i] = (data[:,:,i]-mean_data.iloc[0][i]) / std_data.iloc[0][i]
        elif len(data.shape)==2:
            for i in range(mean_data.shape[1]):
                data[:,i] = (data[:,i]-mean_data.iloc[0][i]) / std_data.iloc[0][i]
    if Scale is False:
        if len(data.shape)==3:
            for i in range(mean_data.shape[1]):
                data[:,:,i] = data[:,:,i]*std_data.iloc[0][i] + mean_data.iloc[0][i]
        elif len(data.shape)==2:
            for i in range(mean_data.shape[1]):
                data[:,i] = data[:,i]*std_data.iloc[0][i] + mean_data.iloc[0][i]
    return data

def minmax_scaler_pd(data, min_data, max_data, Scale):
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    tmp = data.copy()
    if Scale is True:
        for i in data:
            #tmp = data.apply(lambda x: (x - min_data[i])/ (max_data[i] - min_data[i]))
            #data = tmp
            tmp[i] = (data[i]-min_data[i][0]) / (max_data[i][0] - min_data[i][0])
    if Scale is False:
        for i in data:
            #tmp =data.apply(lambda x: (x *(max_data[i] - min_data[i]) + min_data[i]))
            #data = tmp
            tmp[i] = data[i] * (max_data[i][0] - min_data[i][0]) + min_data[i][0]
    return tmp

def minmax_scaler_ndarray(data, min_data, max_data, Scale):
    if Scale is True:
        if len(data.shape)==3:
            for i in range(min_data.shape[1]):
                data[:,:,i] = (data[:,:,i]-min_data.iloc[0][i]) / (max_data.iloc[0][i] - min_data.iloc[0][i])
        elif len(data.shape)==2:
            for i in range(min_data.shape[1]):
                data[:,i] = (data[:,i]-min_data.iloc[0][i]) / (max_data.iloc[0][i] - min_data.iloc[0][i])
    if Scale is False:
        if len(data.shape)==3:
            for i in range(min_data.shape[1]):
                data[:,:,i] = data[:,:,i]*(max_data.iloc[0][i] - min_data.iloc[0][i]) + max_data.iloc[0][i]
        elif len(data.shape)==2:
            for i in range(min_data.shape[1]):
                data[:,i] = data[:,i]*(max_data.iloc[0][i] - min_data.iloc[0][i]) + max_data.iloc[0][i]
    return data

def param_scale(data,type):
    if type =='standardscaler':
        tmp = data.mean()
        mean_data = pd.DataFrame(tmp.values).transpose()
        mean_data.columns = tmp.index
        tmp = data.std()
        std_data = pd.DataFrame(tmp.values).transpose()
        std_data.columns = tmp.index
        for i in std_data:
            if std_data[i].values == 0:
                std_data[i] = 1
        res = {'mean':mean_data,'std':std_data}
    elif type =='MinMaxScaler':
        tmp = data.min()
        min_data = pd.DataFrame(tmp.values).transpose()
        min_data.columns = tmp.index
        tmp = data.max()
        max_data = pd.DataFrame(tmp.values).transpose()
        max_data.columns = tmp.index
        res = {'min':min_data,'max':max_data}
    return res
