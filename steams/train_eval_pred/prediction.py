from typing import Dict
from steams.classes.class_model import class_model
from steams.classes.class_xyv_x import class_xyv_x
from steams.train_eval_pred.trevpr import predict
import os
import pickle

def prediction_function(params:dict):
    params_sessiondir=params['sessiondir']
    params_device = params['device']
    params_data=params['data']
    params_model=params['model']
    sessiondir = params_sessiondir['path']
    model_path = params_model["path"]
    model_name = params_model["filename"]
    if not os.path.exists(sessiondir):
        os.makedirs(sessiondir, exist_ok=True)
    resdir = os.path.join(sessiondir,"pred")
    if not os.path.exists(resdir):
        os.makedirs(resdir, exist_ok=True)
    train_data_filename = os.path.join(model_path,'train','data_class_ts_x.pkl')
    data_filename = os.path.join(resdir,'data_class_ts_x.pkl')
    f_train = open(train_data_filename, 'rb')
    train_data = pickle.load(f_train)
    data = class_xyv_x(params_data)
    data.scale_param_coordinates, data.scale_param_values = train_data.get_scale_param_target()
    data.scale(True)
    model_ = class_model(params_device)
    model_.load(os.path.join(model_path,"train"), model_name)
    output = predict(model_, data)
    print(output)
    #data.df_target = pd.Data.Frame(output.numpy(),columns=data.
    output_unscaled = data.unscale(output,"values")
    prediction_filename = os.path.join(resdir,'prediction.pkl')
    #save pred
    f_pred = open(prediction_filename, 'wb')
    pickle.dump(output_unscaled.numpy(), f_pred)
    # save the class_ts_x data
    f_data = open(data_filename, 'wb')
    pickle.dump(data, f_data)
