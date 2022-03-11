from ml_collections import FrozenConfigDict
from .ebm import EBM
from .loss import loss_KL, loss_ML_KL

__all__ = ["EBM"]
__all__ += ["loss_ML", "loss_ML_KL"]

def get_loss_fn(cfg: FrozenConfigDict):
    return globals()[cfg.EBM.TRAIN.LOSS_NAME]
