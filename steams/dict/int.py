from steams.data.class_xyv_none import class_xyv_none
from steams.models.attention import MW1, MW2, MW3, MW4, sdp, mha
from steams.models.transformer import Transformer, Transformer_ae
from steams.models.train_eval_pred import transformer_steams, transformer_ae_steams
from steams.utils.scheduler import TransformerLRScheduler

class_xyv_dict = {
    "class_xyv_none": class_xyv_none
}

model_dict = {
    "mw1": MW1,
    "mw2": MW2,
    "mw3": MW3,
    "mw4": MW4,
    "sdp": sdp,
    "mha":mha,
    "transformer" : Transformer,
    "transformer_ae" : Transformer_ae
}

steams_dict = {
    "attention": attention_steams,
    "transformer_ae": transformer_ae_steams,
    "transformer": transformer_steams,
}

scheduler_dict = {
    "TransformerLRScheduler": warmup
    }
