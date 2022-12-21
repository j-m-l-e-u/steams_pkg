from typing import Dict
from steams.dict.int import model_dict
import torch

def build_model(params:dict):
    dump = params
    name = params["name"]
    par = params['param']
    if name in model_dict:
        res = model_dict[name](**par)
    else:
        raise Exception("Model " + name +" not found.")
    return(res)


def load_model(path: str, name: str):
    if not os.path.exists(path):
        print("'path' does not exist")
    model_path = os.path.join(path, name + "_model.pth")
    if not os.path.exists(model_path):
        print("'model_path' does not exist")
    params_path = os.path.join(path, name + ".json")
    if not os.path.exists(params_path):
        print("'param_path' does not exist")

    f = open(params_path)
    dump=json.load(f)
    name = dump["name"]
    par = dump['param']
    if name in model_dict:
        model = model_dict[name](**par)
    else:
        raise Exception("Model " + name +" not found.")
    model.load_state_dict(torch.load(model_path))
    return(model)
