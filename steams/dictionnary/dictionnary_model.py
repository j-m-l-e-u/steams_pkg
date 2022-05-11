from steams.models.attention import ED_NW_att, multi_head_att

import torch

model_dict = {
    "ED_NW_att" : ED_NW_att,
    "multi_head_att" : multi_head_att
}
