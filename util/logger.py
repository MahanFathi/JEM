from flax.metrics import tensorboard
from yacs.config import CfgNode

from datetime import datetime
from pathlib import Path

from util.types import *

LOG_PATH = None

def get_logdir_path(cfg: CfgNode) -> Path:
    global LOG_PATH
    if LOG_PATH:
        return LOG_PATH
    logdir_name = "{}_{}".format(
        cfg.ENV.ENV_NAME,
        datetime.now().strftime("%Y.%m.%d_%H:%M:%S"),
    )
    log_path = Path("./logs").joinpath(logdir_name)
    print(log_path)
    log_path.mkdir(parents=True, exist_ok=True)
    LOG_PATH = log_path
    return log_path

def get_summary_writer(cfg: CfgNode) -> tensorboard.SummaryWriter:
    log_path = get_logdir_path(cfg)
    return tensorboard.SummaryWriter(str(log_path))

def save_params(params: Params, name: str, logdir: str = None):
    params_dir = logdir.joinpath("params")
    params_dir.mkdir(exist_ok=True)
    params_file = params_dir.joinpath("{}.flax".format(name))

    param_bytes = flax.serialization.to_bytes(params)

    with open(params_file, "wb") as f:
        f.write(param_bytes)
