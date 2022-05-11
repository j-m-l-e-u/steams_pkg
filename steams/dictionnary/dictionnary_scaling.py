from steams.utils.scale import standard_scaler, minmax_scaler

scaling_dict = {
    "StandardScaler": standard_scaler, #StandardScaler(),
    #"RobustScaler": RobustScaler(),
    "MinMaxScaler": minmax_scaler #MinMaxScaler(),
    #"MaxAbsScaler": MaxAbsScaler()
    }
