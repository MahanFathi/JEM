from typing import Any, Callable, Dict, Optional, Tuple
import functools
import time

from absl import logging
from brax import envs
from brax.training import normalization
from brax.training import pmap
import flax
import jax
import jax.numpy as jnp
import optax
from ml_collections import FrozenConfigDict

from ebm import EBM, get_loss_fn
from envs.base_env import BaseEnv
from envs.base_sampler import BaseSampler
from util.types import *


def train_ebm(
        cfg: FrozenConfigDict,
        env: BaseEnv,
        sampler: BaseSampler,
        progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
        seed: int = 0,
):
    """Training pipeline. Stolen from BRAX for the MOST part."""

    # CONFIG EXTRACTION
    max_devices_per_host = cfg.TRAIN.MAX_DEVICES_PER_HOST
    data_size = cfg.TRAIN.EBM.DATA_SIZE
    num_update_epochs = cfg.TRAIN.EBM.NUM_UPDATE_EPOCHS
    num_samplers = cfg.TRAIN.EBM.NUM_SAMPLERS
    learning_rate = cfg.TRAIN.EBM.LEARNING_RATE
    batch_size = cfg.TRAIN.EBM.BATCH_SIZE
    normalize_observations = cfg.TRAIN.EBM.NORMALIZE_OBSERVATIONS
    log_frequency = cfg.TRAIN.EBM.LOG_FREQUENCY

    # asserts
    xt = time.time()

    # PROCESS BOOKKEEPING
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    logging.info(
        'Device count: %d, process count: %d (id %d), local device count: %d, '
        'devices to be used count: %d',
        jax.device_count(), process_count, process_id, local_device_count,
        local_devices_to_use)

    # KEY MANAGEMENT
    key = jax.random.PRNGKey(seed)
    key, key_model, key_sampler, key_eval = jax.random.split(key, 4)
    # Make sure every process gets a different random key, otherwise they will be
    # doing identical work.
    key_sampler = jax.random.split(key_sampler, process_count)[process_id]
    key = jax.random.split(key, process_count)[process_id]
    # key_models should be the same, so that models are initialized the same way
    # for different processes.
    # key_eval is also used in one process so no need to split.
    keys_sampler = jax.random.split(key_sampler, local_devices_to_use)

    # ENERGY-BASED MODEL
    ebm = EBM(cfg, env)
    init_params = ebm.init(key_model)

    # OPTIMIZER
    optimizer = optax.adam(learning_rate=learning_rate)
    optimizer_state = optimizer.init(init_params)
    optimizer_state, init_params = pmap.bcast_local_devices(
        (optimizer_state, init_params), local_devices_to_use)

    normalizer_params, obs_normalizer_update_fn, obs_normalizer_apply_fn = (
        normalization.create_observation_normalizer(
            env.observation_size, normalize_observations,
            num_leading_batch_dims=2, pmap_to_devices=local_devices_to_use))

    key_debug = jax.random.PRNGKey(seed + 666)

    # LOSS AND GRAD
    loss_fn = functools.partial(
        get_loss_fn(cfg.TRAIN.LOSS_FN),
        cfg=cfg, ebm=ebm,
    )
    grad_loss = jax.grad(loss_fn, has_aux=True)

