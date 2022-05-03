from ml_collections import FrozenConfigDict
from .net import make_ebm_model_arch0, make_ebm_model_arch1

__all__ = []
__all__ += ["make_ebm_model_arch0"]
__all__ += ["make_ebm_model_arch1"]

_registry = {
    "ebm": {
        "arch0": make_ebm_model_arch0,
        "arch1": make_ebm_model_arch1,
    },
}

def build_ebm_net(cfg: FrozenConfigDict, observation_size: int, action_size: int):
    return _registry["ebm"][cfg.EBM.ARCH](cfg, observation_size, action_size)
