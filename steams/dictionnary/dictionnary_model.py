from steams.models.attention import MLP_NW_dist_att, MLP_NW_dist_2_att ,multi_head_att

import torch

model_dict = {
    "MLP_NW_dist_att" : MLP_NW_dist_att,
    "MLP_NW_dist_2_att" : MLP_NW_dist_2_att,
    "multi_head_att" : multi_head_att
}
