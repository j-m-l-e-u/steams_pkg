from steams.data.class_xyv_none import class_xyv_none
from steams.models.transformer import Transformer, Transformer_ae
from steams.models.train_eval_pred import transformer_steams, transformer_ae_steams
from steams.utils.scheduler import TransformerLRScheduler

class_xyv_dict = {
    "class_xyv_none": class_xyv_none
}

model_dict = {
    "transformer" : Transformer,
    "transformer_ae" : Transformer_ae
}

steams_dict = {
    "transformer": transformer_steams,
    "transformer_ae": transformer_ae_steams,
}

scheduler_dict = {
    "TransformerLRScheduler": warmup
    }
