from typing import Tuple
from functools import partial

import jax
from jax import numpy as jnp
from ml_collections import FrozenConfigDict

from ebm import EBM
from ebm.ebm import infer_z_then_a, infer_z
from util.types import *



@partial(jax.vmap, in_axes=(0, None, None))
def _calc_action_distance_batched_a(action_pred, action, discount):
    # TODO: either support normalized action, or remove it altogether
    return jnp.mean(discount ** jnp.arange(action.shape[1]) * jnp.linalg.norm(action - action_pred, axis=-1))


def _calc_action_distance(action_pred, action, discount):
    # TODO: either support normalized action, or remove it altogether
    return jnp.mean(discount ** jnp.arange(action.shape[1]) * jnp.linalg.norm(action - action_pred, axis=-1))


def _calc_loss_contrastive(params: Params, data: StepData, key: PRNGKey, cfg: FrozenConfigDict, ebm: EBM):

    batch_size, horizon, action_size = data.action.shape

    key, key_infer_z = jax.random.split(key)
    z = infer_z(params, data, key_infer_z, cfg, ebm)
    z = jnp.stack([z] * (horizon - 1), axis=1) # (batch_size, horizon - 1, option_size)

    key, key_sample_a = jax.random.split(key)
    a_sample_size = cfg.TRAIN.EBM.NEGATIVE_ACTION_SAMPLE_SIZE
    negative_a = jax.random.uniform(key_sample_a, (a_sample_size, batch_size, horizon - 1, action_size)) # TODO: assuming actions lie in range [0, 1]

    # NOTE: we allow gradient to flow inside z(\theta)
    loss_contrastive_positive = ebm.apply(params, data.observation[:, 1:, :], z, data.action[:, 1:, :])
    loss_contrastive_negative = ebm.apply_batch_a(params, data.observation[:, 1:, :], z, negative_a).mean(axis=0)

    loss_contrastive = loss_contrastive_positive - loss_contrastive_negative # TODO: optionally add softmax
    loss_contrastive = loss_contrastive.mean()

    return {
        "loss_contrastive": loss_contrastive,
    }


def _calc_loss_ml_kl_l2(params: Params, data: StepData, key: PRNGKey, cfg: FrozenConfigDict, ebm: EBM):
    # infer z and a
    #   z: (batch_size, option_size)
    #   a: (action_infer_batch_size, batch_size, horizon - 1, action_size)
    key, key_infer = jax.random.split(key)
    z, a = infer_z_then_a(params, data, key, cfg, ebm)

    horizon = a.shape[-2] + 1
    z = jnp.stack([z] * (horizon - 1), axis=1) # (batch_size, horizon - 1, option_size)

    # calc loss ml
    discount = cfg.TRAIN.EBM.DISCOUNT
    loss_ml = discount ** jnp.arange(horizon - 1) * jax.nn.softplus(
        ebm.apply(params, data.observation[:, 1:, :], z, data.action[:, 1:, :]) -
        ebm.apply_batch_a(params, data.observation[:, 1:, :], z, jax.lax.stop_gradient(a)).mean(axis=0)
    ).mean(axis=0)
    loss_ml = loss_ml.mean()

    # calc loss kl
    loss_kl = discount ** jnp.arange(horizon - 1) * ebm.apply_batch_a(
        jax.lax.stop_gradient(params), data.observation[:, 1:, :],
        jax.lax.stop_gradient(z), a).mean(axis=(0, 1))
    loss_kl = loss_kl.mean()

    loss_l2 = _calc_action_distance_batched_a(a, data.action[:, 1:, :], discount).mean()

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
    _, a = infer_z_then_a(params, data, key, cfg, ebm)

    loss_l2 = _calc_action_distance_batched_a(a, data.action[:, 1:, :], cfg.TRAIN.EBM.DISCOUNT).mean()
    return loss_l2, {
        "loss": loss_l2,
        "loss_l2": loss_l2,
    }


def loss_contrastive(params: Params, data: StepData, key: PRNGKey, cfg: FrozenConfigDict, ebm: EBM):
    losses = _calc_loss_contrastive(params, data, key, cfg, ebm)
    loss_contrastive = losses["loss_contrastive"]
    return loss_contrastive, {**{"loss": loss_contrastive}, **losses}


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
    loss_kl = losses["loss_kl"]
    loss = loss_l2 + cfg.TRAIN.EBM.LOSS_KL_COEFF * loss_kl
    return loss, {**{"loss": loss}, **losses}


# EVAL FNs
def eval_action_l2(params: Params, data: StepData, key: PRNGKey, cfg: FrozenConfigDict, ebm: EBM):
    # infer z and a
    #   z: (batch_size, option_size)
    #   a: (batch_size, horizon - 1, action_size)
    key, key_infer_z = jax.random.split(key)

    batch_size, horizon, _ = data.observation.shape

    z = infer_z(params, data, key_infer_z, cfg, ebm, langevin_gd=False)
    stacked_z = jnp.stack((horizon - 1) * [z]).swapaxes(0, 1) # (batch_size, horizon - 1, option_size)

    a = ebm.infer_batch_a_derivative_free(params, data.observation[:, 1:, :], stacked_z, jax.random.split(key, batch_size)) # (batch_size, horizon - 1, action_size)

    action_l2 = _calc_action_distance(a, data.action[:, 1:, :], 1.0).mean()
    action_l2_discounted = _calc_action_distance(a, data.action[:, 1:, :], cfg.TRAIN.EBM.DISCOUNT).mean()
    return {
        "eval_action_l2": action_l2,
        "eval_action_l2_discounted": action_l2_discounted,
    }
