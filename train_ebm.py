from typing import Any, Callable, Dict, Optional, Tuple
import functools
import time

from absl import logging
from brax import envs
from brax.training import distribution
from brax.training import normalization
from brax.training import pmap
import flax
import jax
import jax.numpy as jnp
import optax
from ml_collections import FrozenConfigDict

from ebm import EBM, get_loss_fn, eval_action_l2
from envs.base_env import BaseEnv
from envs.base_sampler import BaseSampler
from util.types import *


def train_ebm(
        cfg: FrozenConfigDict,
        env: BaseEnv,
        sampler: BaseSampler,
        key: PRNGKey,
        progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
):
    """Training pipeline. Stolen from BRAX for the MOST part."""

    # CONFIG EXTRACTION
    max_devices_per_host = cfg.TRAIN.MAX_DEVICES_PER_HOST
    data_size = cfg.TRAIN.EBM.DATA_SIZE
    num_epochs = cfg.TRAIN.EBM.NUM_EPOCHS
    num_update_epochs = cfg.TRAIN.EBM.NUM_UPDATE_EPOCHS
    num_samplers = cfg.TRAIN.EBM.NUM_SAMPLERS
    learning_rate = cfg.TRAIN.EBM.LEARNING_RATE
    batch_size = cfg.TRAIN.EBM.BATCH_SIZE
    eval_batch_size = cfg.TRAIN.EBM.EVAL_BATCH_SIZE
    num_minibatches = cfg.TRAIN.EBM.NUM_MINIBATCHES
    normalize_observations = cfg.TRAIN.EBM.NORMALIZE_OBSERVATIONS
    normalize_actions = cfg.TRAIN.EBM.NORMALIZE_ACTIONS # TODO: support normalization w/ fixed params
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
    key, key_model, key_eval = jax.random.split(key, 3)
    # Make sure every process gets a different random key, otherwise they will be
    # doing identical work.
    key = jax.random.split(key, process_count)[process_id]
    # key_models should be the same, so that models are initialized the same way
    # for different processes.
    # key_eval is also used in one process so no need to split.

    # ENERGY-BASED MODEL
    ebm = EBM(cfg, env)
    init_params = ebm.init(key_model)

    # OPTIMIZER
    optimizer = optax.adam(learning_rate=learning_rate)
    optimizer_state = optimizer.init(init_params)
    optimizer_state, init_params = pmap.bcast_local_devices(
        (optimizer_state, init_params), local_devices_to_use)

    # NORMALIZER
    obs_normalizer_params, obs_normalizer_update_fn, obs_normalizer_apply_fn = (
        normalization.create_observation_normalizer(
            env.observation_size, normalize_observations,
            num_leading_batch_dims=2, pmap_to_devices=local_devices_to_use))
    # weird cuz action normalizer is also instantiated from
    # create_observation_normalizer, but this is on BRAX for shitty naming
    act_normalizer_params, act_normalizer_update_fn, act_normalizer_apply_fn = (
        normalization.create_observation_normalizer(
            env.action_size, normalize_actions,
            num_leading_batch_dims=2, pmap_to_devices=local_devices_to_use))

    # LOSS AND GRAD
    loss_fn = functools.partial(
        get_loss_fn(cfg.TRAIN.EBM.LOSS_NAME),
        cfg=cfg, ebm=ebm,
    )
    grad_loss = jax.grad(loss_fn, has_aux=True)


    def update_model(carry, data):
        optimizer_state, params, key = carry
        key_loss, key = jax.random.split(key)
        loss_grad, metrics = grad_loss(params, data, key_loss)
        loss_grad = jax.lax.pmean(loss_grad, axis_name='i')
        params_update, optimizer_state = optimizer.update(
            loss_grad,
            optimizer_state,
        )
        params = optax.apply_updates(params, params_update)
        return (optimizer_state, params, key), metrics

    def minimize_epoch(carry, unused_t):
        optimizer_state, params, data, key = carry
        key_grad, key = jax.random.split(key)

        ndata = jax.tree_map(
            lambda x: x.reshape(num_minibatches, -1, x.shape[1], *x.shape[2:]),
            data) # TODO: is this really necessary?
        # data: (num_minibatches, batch_size / num_minibatches, horizon, dim)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            update_model, (optimizer_state, params, key_grad),
            ndata,
            length=num_minibatches)
        return (optimizer_state, params, data, key), metrics

    def run_epoch(carry: Tuple[TrainingState, ], unused_t):
        training_state, = carry
        key_sampler, key_minimize, key = jax.random.split(
            training_state.key, 3)

        # sample data (batch, horizon, dim)
        data: StepData = sampler.sample_batch_subtrajectory(
            batch_size // local_device_count, key_sampler)

        # Update normalization params and normalize observations/actions.
        obs_normalizer_params = obs_normalizer_update_fn(
            training_state.obs_normalizer_params, data.observation)
        act_normalizer_params = act_normalizer_update_fn(
            training_state.act_normalizer_params, data.action)
        data = data.replace(
            observation=obs_normalizer_apply_fn(obs_normalizer_params, data.observation),
            action=act_normalizer_apply_fn(act_normalizer_params, data.action),
        )

        (optimizer_state, params, _, _), metrics = jax.lax.scan(
            minimize_epoch, (training_state.optimizer_state, training_state.params,
                             data, key_minimize), (),
            length=num_update_epochs)

        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            obs_normalizer_params=obs_normalizer_params,
            act_normalizer_params=act_normalizer_params,
            key=key)
        return (new_training_state, ), metrics


    def _minimize_loop(training_state):
        synchro = pmap.is_replicated(
            (training_state.optimizer_state, training_state.params,
             training_state.obs_normalizer_params),
            axis_name='i')
        (training_state, ), losses = jax.lax.scan(
            run_epoch, (training_state, ), (),
            length=num_epochs // log_frequency)
        losses = jax.tree_map(jnp.mean, losses)
        return (training_state, ), losses, synchro


    minimize_loop = jax.pmap(_minimize_loop, axis_name='i')

    training_state = TrainingState(
        optimizer_state=optimizer_state,
        params=init_params,
        key=jnp.stack(jax.random.split(key, local_devices_to_use)),
        obs_normalizer_params=obs_normalizer_params,
        act_normalizer_params=act_normalizer_params)

    training_walltime = 0
    sps = 0
    losses = {}
    metrics = {}

    for it in range(log_frequency + 1):
        t = time.time()

        if process_id == 0:

            # do eval
            key_eval, key_eval_infer, key_eval_sampler = jax.random.split(key_eval, 3)
            eval_data: StepData = sampler.sample_batch_subtrajectory(
                batch_size, key_eval_sampler)
            eval_data = eval_data.replace(
                observation=obs_normalizer_apply_fn(obs_normalizer_params, eval_data.observation),
                action=act_normalizer_apply_fn(act_normalizer_params, eval_data.action),
            )
            eval_params = jax.tree_map(lambda x: x[0], training_state.params)
            evals = eval_action_l2(eval_params, eval_data, key_eval_infer, cfg, ebm)

            # log callback
            metrics = {**losses, **evals}
            logging.info('starting iteration %s %s', it, time.time() - xt)
            logging.info('metrics: {}'.format(metrics))
            if progress_fn:
                progress_fn(int(training_state.obs_normalizer_params[0][0]), metrics)

        if it == log_frequency:
            break

        t = time.time()
        previous_step = training_state.obs_normalizer_params[0][0]
        # optimization
        (training_state, ), losses, synchro = minimize_loop(training_state)
        logging.log(logging.DEBUG, "\n***TrainingState***: \n{}".format(training_state))
        assert synchro[0], (it, training_state)
        jax.tree_map(lambda x: x.block_until_ready(), losses)
        sps = ((training_state.obs_normalizer_params[0][0] - previous_step) /
               (time.time() - t))
        training_walltime += time.time() - t

    # To undo the pmap.
    obs_normalizer_params = jax.tree_map(lambda x: x[0],
                                         training_state.obs_normalizer_params)
    act_normalizer_params = jax.tree_map(lambda x: x[0],
                                         training_state.act_normalizer_params)
    ebm_params = jax.tree_map(lambda x: x[0], training_state.params)

    logging.info('total steps: %s', obs_normalizer_params[0])

    pmap.synchronize_hosts()

    return 0
