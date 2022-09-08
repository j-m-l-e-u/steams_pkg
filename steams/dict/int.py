from steams.data.KVyQVx import KVyQVx
from steams.models.attention import NW0,NW1, NW2, NW3, NW4, NW5, NW6, mha0, mha1, mha2
from steams.models.krig import krignn, krignn2
from steams.models.transformer import Transformer_coords, Transformer_ae
from steams.models.train_eval_pred import attention_steams, transformer_ae_steams, transformer_coords_steams,transformer_steams
from steams.utils.scheduler import TransformerLRScheduler

KVyQVx_dict = {
    "KVyQVx": KVyQVx
}

model_dict = {
    "NW0": NW0,
    "NW1": NW1,
    "NW2": NW2,
    "NW3": NW3,
    "NW4": NW4,
    "NW5": NW5,
    "NW6": NW6,
    "NW7": NW7,
    "mha0":mha0,
    "mha1":mha1,
    "mha2":mha2,
    "krignn":krignn,
    "krignn2":krignn2,
    "transformer_coords" : Transformer_coords,
    "transformer_ae" : Transformer_ae,
    "NW0mer_ae": NW0mer_ae
}

steams_dict = {
    "attention": attention_steams,
    "nwmer_ae": nwmer_ae_steams,
    "transformer": transformer_steams,
    "transformer_coords": transformer_coords_steams
}

scheduler_dict = {
    "TransformerLRScheduler": TransformerLRScheduler
    }
