from typing import Tuple

import jax
from jax import numpy as jnp
from ml_collections import FrozenConfigDict

from ebm import EBM
from ebm.ebm import infer_z_and_a
from util.types import *


# helper functions
def _soft_plus(x):
    return jnp.log(1. + jnp.exp(x))


def _calc_action_distance(a, a_pred, discount):
    # TODO: either support normalized action, or remove it altogether
    return jnp.mean(discount ** jnp.arange(a.shape[1]) * jnp.linalg.norm(a - a_pred, axis=-1))


def _calc_loss_ml_kl_l2(params: Params, data: StepData, key: PRNGKey, cfg: FrozenConfigDict, ebm: EBM):

    # infer z and a
    #   z: (batch_size, option_size)
    #   a: (batch_size, horizon - 1, action_size)
    key, key_infer = jax.random.split(key)
    z, a = infer_z_and_a(params, data, key, cfg, ebm)

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

    loss_l2 = _calc_action_distance(data.action[:, 1:, :], a, discount)

    return {
        "loss_ml": loss_ml,
        "loss_kl": loss_kl,
        "loss_l2": loss_l2,
    }


# LOSS FNs
def loss_L2(params: Params, data: StepData, key: PRNGKey, cfg: FrozenConfigDict, ebm: EBM):
    # infer z and a
    #   z: (batch_size, option_size)
    #   a: (batch_size, horizon - 1, action_size)
    key, key_infer = jax.random.split(key)
    _, a = infer_z_and_a(params, data, key, cfg, ebm)

    loss_l2 = _calc_action_distance(data.action[:, 1:, :], a, cfg.TRAIN.EBM.DISCOUNT)
    return loss_l2, {
        "loss": loss_l2,
        "loss_l2": loss_l2,
    }


def loss_ML(params: Params, data: StepData, key: PRNGKey, cfg: FrozenConfigDict, ebm: EBM):
    losses = _calc_loss_ml_kl_l2(params, data, key, cfg, ebm) # TODO: fix, not efficient
    loss_ml = losses["loss_ml"]
    return loss_ml, {**{"loss": loss_ml}, **losses}


def loss_ML_KL(params: Params, data: StepData, key: PRNGKey, cfg: FrozenConfigDict, ebm: EBM):
    losses = _calc_loss_ml_kl_l2(params, data, key, cfg, ebm)
    loss_ml = losses["loss_ml"]
    loss_kl = losses["loss_kl"]
    loss = loss_ml + cfg.TRAIN.EBM.LOSS_KL_COEFF * loss_kl
    return loss, {**{"loss": loss}, **losses}


def loss_L2_KL(params: Params, data: StepData, key: PRNGKey, cfg: FrozenConfigDict, ebm: EBM):
    losses = _calc_loss_ml_kl_l2(params, data, key, cfg, ebm)
    loss_l2 = losses["loss_l2"]
    loss = loss_l2 + cfg.TRAIN.EBM.LOSS_KL_COEFF * loss_kl
    return loss, {**{"loss": loss}, **losses}


# EVAL FNs
def eval_action_l2(params: Params, data: StepData, key: PRNGKey, cfg: FrozenConfigDict, ebm: EBM):
    # infer z and a
    #   z: (batch_size, option_size)
    #   a: (batch_size, horizon - 1, action_size)
    key, key_infer = jax.random.split(key)
    _, a = infer_z_and_a(params, data, key, cfg, ebm, langevin_gd=False)

    action_l2 = _calc_action_distance(data.action[:, 1:, :], a, 1.0)
    action_l2_discounted = _calc_action_distance(data.action[:, 1:, :], a, cfg.TRAIN.EBM.DISCOUNT)
    return {
        "eval_action_l2": action_l2,
        "eval_action_l2_discounted": action_l2_discounted,
    }
