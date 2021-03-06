from .ebm import EBM
from .loss import loss_ML, loss_ML_KL, loss_L2, loss_L2_KL, loss_contrastive, loss_ML_KL_KLz
from .loss import eval_action_l2

__all__ = ["EBM"]
__all__ += [
    "loss_ML",
    "loss_ML_KL",
    "loss_L2",
    "loss_L2_KL",
    "loss_contrastive",
    "loss_ML_KL_KLz"
]
__all__ += ["eval_action_l2"]

def get_loss_fn(loss_name: str):
    return globals()[loss_name]
