from typing import Dict
from datetime import datetime
from steams.dictionnary.dictionnary_model import model_dict
import torch
import os
import json

class class_model():
    def __init__(self, params: dict):
        # parameters related to the device (cpu,cuda)
        if params['cuda'] is not None and torch.cuda.is_available():
            self.device = torch.device('cuda'+":"+params['cuda']['name'])
        elif params['cpu'] is not None:
            self.device = torch.device('cpu')
        else:
            raise Exception("device not specified.")
        print("Torch is using " + str(self.device))

    def build(self,params:dict):
        self.dump = params
        self.name = params["name"]
        self.params = params['param']
        if self.name in model_dict:
            self.model = model_dict[self.name](self.device,**self.params)
            self.model.to(self.device)
        else:
            raise Exception("Model " + self.name +" not found.")

    def load(self, path: str, name: str):
        if not os.path.exists(path):
            print("'path' does not exist")
        model_path = os.path.join(path, name + "_model.pth")
        if not os.path.exists(model_path):
            print("'model_path' does not exist")
        params_path = os.path.join(path, name + ".json")
        if not os.path.exists(params_path):
            print("'param_path' does not exist")
        f = open(params_path)
        self.dump=json.load(f)
        self.name = self.dump["name"]
        self.params = self.dump['param']
        if self.name in model_dict:
            self.model = model_dict[self.name](self.device,**self.params)
            self.model.to(self.device)
        else:
            raise Exception("Model " + self.name +" not found.")
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def save(self, path: str, name:str) -> None:
        if not os.path.exists(path):
            os.mkdir(path)
        model_path = os.path.join(path, name + "_model.pth")
        params_path = os.path.join(path, name + ".json")
        torch.save(self.model.state_dict(), model_path)
        with open(params_path, "w+") as p:
            json.dump(self.dump, p)

    def saveCheckpoint(self,path: str, name:str, epoch, opt, loss,index=None):
        if not os.path.exists(path):
            os.mkdir(path)
        checkpoint_files = [f for f in os.listdir(path) if f.endswith('_checkpoint.pth')]
        if len(checkpoint_files)==10:
            for file in checkpoint_files:
                os.remove(os.path.join(path, file))
        checkpoint_path = os.path.join(path, name + "_checkpoint.pth")
        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'loss': loss,
                    'index': index}, checkpoint_path)

    def export_onnx(self, path: str, name:str, class_xyv_,params:dict):
        if params is not None:
            if params['param']['opset_version'] is not None:
                opset_version = params['param']['opset_version']
            else:
                opset_version = 9
            # further params ...
            if not os.path.exists(path):
                os.mkdir(path)
            model_path = os.path.join(path, name + "_model.onnx")
            #torch.onnx.export(self.model, class_xyv_.get_rand_input().to(self.device), model_path,opset_version=opset_version)
            torch.onnx.export(self.model, class_xyv_.get_rand_input(), model_path,opset_version=opset_version)
