from typing import Dict
from steams.dict.int import class_xyv_dict

def class_xyv_x(params: dict,subset_indice=None):
    class_x = class_xyv_dict[params['class']]
    res = class_x(params,subset_indice)
    return(res)
