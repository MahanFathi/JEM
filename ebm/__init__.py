from .ebm import EBM
from .loss import loss_ML, loss_ML_KL, loss_L2, loss_L2_KL

__all__ = ["EBM"]
__all__ += ["loss_ML", "loss_ML_KL", "loss_L2", "loss_L2_KL"]

def get_loss_fn(loss_name: str):
    return globals()[loss_name]
