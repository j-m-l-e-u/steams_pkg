from steams.data.class_xyv_none import class_xyv_none
from steams.models.attention import MW0,MW1, MW2, MW3, MW4, MW5, MW6, mha, mha2
from steams.models.krig import krignn, krignn2
from steams.models.transformer import Transformer, Transformer_ae, Transformer_coords
from steams.models.train_eval_pred import transformer_steams, transformer_ae_steams, transformer_coords_steams
from steams.utils.scheduler import TransformerLRScheduler

class_xyv_dict = {
    "class_xyv_none": class_xyv_none
}

model_dict = {
    "mw0": MW0,
    "mw1": MW1,
    "mw2": MW2,
    "mw3": MW3,
    "mw4": MW4,
    "mw5": MW5,
    "mw6": MW6,
    "mha0":mha0,
    "mha1":mha1,
    "mha2":mha2,
    "krignn":krignn,
    "krignn2":krignn2,
    "transformer" : Transformer,
    "transformer_ae" : Transformer_ae,
    "transformer_coords": Transformer_coords
}

steams_dict = {
    "attention": attention_steams,
    "transformer_ae": transformer_ae_steams,
    "transformer": transformer_steams,
    "transformer_coords": transformer_coords_steams
}

scheduler_dict = {
    "TransformerLRScheduler": warmup
    }
