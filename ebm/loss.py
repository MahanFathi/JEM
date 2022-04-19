from typing import Tuple

import jax
from jax import numpy as jnp
from ml_collections import FrozenConfigDict

from ebm import EBM
from util.types import *

# from absl import logging
# from jax.experimental import host_callback as jhcb


# helper functions
def _soft_plus(x):
    return jnp.log(1. + jnp.exp(x))


def _infer_z_and_a(params: Params, data: StepData, key: PRNGKey, cfg: FrozenConfigDict, ebm: EBM):
    """
    Infers the option from the first transition in the data and
    then predicts future actions based on the inferred option.

    data: (batch_size, horizon, dim)

    returns:
        z: (batch_size, option_size)
        a: (batch_size, horizon - 1, action_size)
    """

    batch_size, horizon, action_size = data.action.shape
    option_size = cfg.EBM.OPTION_SIZE

    # infer z from first state-action
    key, key_init_z, key_infer_z = jax.random.split(key, 3)
    z_init = jax.random.normal(key_init_z, (batch_size, option_size))
    z = ebm.infer_z(
        params, data.observation[:, 0, :],
        z_init, data.action[:, 0, :], key_infer_z,
    )

    # infer actions based on the inferred option
    key, key_init_a, key_infer_a = jax.random.split(key, 3)
    a_init = jax.random.normal(key_init_a, (horizon - 1, batch_size, action_size))
    _, a = jax.lax.scan(
        ebm.scan_to_infer_multiple_a, (params, z, key),
        StepData(
            observation=data.observation.swapaxes(0, 1)[1:],
            action=a_init,
        ),
    )

    a = a.swapaxes(0, 1)

    # if cfg.DEBUG:
        # debug_log_fn = lambda x: logging.log(logging.DEBUG, "\n***Inferred Actions***: \n{}".format(x))
        # jhcb.id_tap(debug_log_fn, a)
        # debug_log_fn = lambda x: logging.log(logging.DEBUG, "\n***Inferred Options***: \n{}".format(x))
        # jhcb.id_tap(debug_log_fn, z)

        # print("\ninferred options: \n")
        # jhcb.id_tap(print, z)
        # print("\ninferred actions: \n")
        # jhcb.id_tap(print, a)

    return z, a


def _calc_action_distance(a, a_pred, discount):
    return jnp.mean(discount ** jnp.arange(a.shape[1]) * jnp.linalg.norm((a - a_pred)))

def _cal_loss_ml_kl(params: Params, data: StepData, key: PRNGKey, cfg: FrozenConfigDict, ebm: EBM):

    # infer z and a
    #   z: (batch_size, option_size)
    #   a: (batch_size, horizon - 1, action_size)
    key, key_infer = jax.random.split(key)
    z, a = _infer_z_and_a(params, data, key, cfg, ebm)

    horizon = a.shape[1] + 1
    z = jnp.stack([z] * (horizon - 1), axis=1) # (batch_size, horizon - 1, option_size)

    # calc loss ml
    discount = cfg.TRAIN.EBM.DISCOUNT
    loss_ml = discount ** jnp.arange(horizon - 1) * _soft_plus(
        ebm.apply(params, data.observation[:, 1:, :], z, data.action[:, 1:, :]) -
        ebm.apply(params, data.observation[:, 1:, :], z, jax.lax.stop_gradient(a))
    ).mean(axis=0)
    loss_ml = loss_ml.mean()

    # calc loss kl
    loss_kl = discount ** jnp.arange(horizon - 1) * ebm.apply(
        jax.lax.stop_gradient(params), data.observation[:, 1:, :],
        jax.lax.stop_gradient(z), a).mean(axis=0)
    loss_kl = loss_kl.mean()

    aux = {"loss_l2": _calc_action_distance(data.action[:, 1:, :], a, discount)}

    return loss_ml, loss_kl, aux


def loss_L2(params: Params, data: StepData, key: PRNGKey, cfg: FrozenConfigDict, ebm: EBM):

    # infer z and a
    #   z: (batch_size, option_size)
    #   a: (batch_size, horizon - 1, action_size)
    key, key_infer = jax.random.split(key)
    _, a = _infer_z_and_a(params, data, key, cfg, ebm)

    loss_l2 = _calc_action_distance(data.action[:, 1:, :], a, discount)

    return loss_l2, {
        "loss": loss_l2,
        "loss_l2": loss_l2,
    }


def loss_ML(params: Params, data: StepData, key: PRNGKey, cfg: FrozenConfigDict, ebm: EBM):
    loss_ml, _, aux = _cal_loss_ml_kl(params, data, key, cfg, ebm) # TODO: fix, not efficient
    return loss_ml, {**{
        "loss": loss_ml,
        "loss_ml": loss_ml,
    }, **aux}


def loss_ML_KL(params: Params, data: StepData, key: PRNGKey, cfg: FrozenConfigDict, ebm: EBM):
    loss_ml, loss_kl, aux = _cal_loss_ml_kl(params, data, key, cfg, ebm)
    loss = loss_ml + cfg.TRAIN.EBM.LOSS_KL_COEFF * loss_kl
    return loss, {**{
        "loss": loss,
        "loss_ml": loss_ml,
        "loss_kl": loss_kl,
    }, **aux}

