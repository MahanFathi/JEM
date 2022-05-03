from functools import partial

import jax
from jax import numpy as jnp
from ml_collections import FrozenConfigDict

from envs.base_env import BaseEnv
from util.net import build_ebm_net
from util.types import *

# from absl import logging
# from jax.experimental import host_callback as jhcb


class EBM(object):

    def __init__(self, cfg: FrozenConfigDict, env: BaseEnv, key: PRNGKey = None):

        self.cfg = cfg
        self.env = env

        self.state_size = env._observation_size
        self.action_size = env._action_size
        self.option_size = cfg.EBM.OPTION_SIZE

        # build net
        self._ebm_net = build_ebm_net(cfg, self.state_size, self.action_size)

        # define derivatives
        self._dedz = jax.jit(jax.vmap(jax.grad(self.apply, 2), in_axes=(None, 0, 0, 0)))
        self._deda = jax.jit(jax.vmap(jax.grad(self.apply, 3), in_axes=(None, 0, 0, 0)))

        if cfg.EBM.GRAD_CLIP:
            self.dedz = lambda params, s, z, a: jnp.clip(
                self._dedz(params, s, z, a), -cfg.EBM.GRAD_CLIP, cfg.EBM.GRAD_CLIP)
            self.deda = lambda params, s, z, a: jnp.clip(
                self._deda(params, s, z, a), -cfg.EBM.GRAD_CLIP, cfg.EBM.GRAD_CLIP)
        else:
            self.dedz = self._dedz
            self.deda = self._deda


    def init(self, key: PRNGKey):
        return self._ebm_net.init(key)


    @partial(jax.jit, static_argnums=(0,))
    def apply(self, params: Params, s: jnp.ndarray, z: jnp.ndarray, a: jnp.ndarray):
        return self._ebm_net.apply(params, s, z, a).squeeze() ** 2 # (batch_size, 1).squeeze()


    @partial(jax.jit, static_argnums=(0, ))
    def _step_z_grad_descent(self, carry, unused_t):
        params: Params
        s: jnp.ndarray
        z: jnp.ndarray
        a: jnp.ndarray
        key: PRNGKey
        langevin_gd: bool
        params, s, z, a, key, langevin_gd = carry

        alpha = self.cfg.EBM.ALPHA

        key, langevin_key = jax.random.split(key)
        omega = jax.random.normal(langevin_key, z.shape) * jnp.sqrt(alpha)
        omega *= langevin_gd # TODO: dirty way around jax compiling scan functions
        # TODO: do not add noise at final step?

        z += -alpha / 2. * self.dedz(params, s, z, a) + omega

        return (params, s, z, a, key, langevin_gd), ()


    @partial(jax.jit, static_argnums=(0, ))
    def _step_a_grad_descent(self, carry, unused_t):
        params: Params
        s: jnp.ndarray
        z: jnp.ndarray
        a: jnp.ndarray
        key: PRNGKey
        langevin_gd: bool
        params, s, z, a, key, langevin_gd = carry

        alpha = self.cfg.EBM.ALPHA

        key, langevin_key = jax.random.split(key)
        omega = jax.random.normal(langevin_key, a.shape) * jnp.sqrt(alpha)
        omega *= langevin_gd # TODO: dirty way around jax compiling scan functions
        # TODO: do not add noise at final step?

        a += -alpha / 2. * self.deda(params, s, z, a) + omega

        return (params, s, z, a, key, langevin_gd), ()


    @partial(jax.jit, static_argnums=(0, ))
    def _step_z_and_a_grad_descent(self, carry, unused_t):
        params: Params
        s: jnp.ndarray
        z: jnp.ndarray
        a: jnp.ndarray
        key: PRNGKey
        langevin_gd: bool
        params, s, z, a, key, langevin_gd = carry

        alpha = self.cfg.EBM.ALPHA # NOTE: here we assume we have used same alpha for z and a during training

        key, langevin_key_z, langevin_key_a = jax.random.split(key, 3)
        omega_z = jax.random.normal(langevin_key_z, z.shape) * jnp.sqrt(alpha)
        omega_a = jax.random.normal(langevin_key_a, a.shape) * jnp.sqrt(alpha)
        omega_z *= langevin_gd # TODO: dirty way around jax compiling scan functions
        omega_a *= langevin_gd # TODO: dirty way around jax compiling scan functions
        # TODO: do not add noise at final step?

        dedz = self.dedz(params, s, z, a)
        deda = self.deda(params, s, z, a)
        z += -alpha / 2. * dedz + omega_z
        a += -alpha / 2. * deda + omega_a

        return (params, s, z, a, key, langevin_gd), ()


    def infer_z(self, params: Params, s: jnp.ndarray, z: jnp.ndarray, a: jnp.ndarray, key: PRNGKey, langevin_gd: bool = None):

        if langevin_gd is None:
            langevin_gd = self.cfg.EBM.LANGEVIN_GD

        (_, _, z, _, _, _), _ = jax.lax.scan(
            self._step_z_grad_descent,
            (params, s, z, a, key, langevin_gd), (), self.cfg.EBM.K)

        return z


    def infer_a(self, params: Params, s: jnp.ndarray, z: jnp.ndarray, a: jnp.ndarray, key: PRNGKey, langevin_gd: bool = None):

        if langevin_gd is None:
            langevin_gd = self.cfg.EBM.LANGEVIN_GD

        (_, _, _, a, _, _), _ = jax.lax.scan(
            self._step_a_grad_descent,
            (params, s, z, a, key, langevin_gd), (), self.cfg.EBM.K)

        return a


    def infer_z_and_a(self, params: Params, s: jnp.ndarray, z: jnp.ndarray, a: jnp.ndarray, key: PRNGKey, langevin_gd: bool = None):

        if langevin_gd is None:
            langevin_gd = self.cfg.EBM.LANGEVIN_GD

        (_, _, z, a, _, _), _ = jax.lax.scan(
            self._step_z_and_a_grad_descent,
            (params, s, z, a, key, langevin_gd), (), self.cfg.EBM.K)

        return z, a


    def scan_to_infer_multiple_a(self, carry, x: StepData):
        params, z, key, langevin_gd = carry
        s = x.observation
        a_init = x.action

        key, key_infer_a = jax.random.split(key)

        a = self.infer_a(params, s, z, a_init, key_infer_a, langevin_gd)

        return (params, z, key, langevin_gd), a



def infer_z_then_a(params: Params, data: StepData, key: PRNGKey, cfg: FrozenConfigDict, ebm: EBM, langevin_gd: bool = None):
    """
    Infers the option from the first transition in the data and
    then predicts future actions based on the inferred option.

    data: (batch_size, horizon, dim)

    returns:
        z: (batch_size, option_size)
        a: (batch_size, horizon - 1, action_size)
    """

    if langevin_gd is None:
        langevin_gd = cfg.EBM.LANGEVIN_GD

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
        ebm.scan_to_infer_multiple_a, (params, z, key, langevin_gd),
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


def infer_z_and_a(params: Params, s: jnp.ndarray, key: PRNGKey, cfg: FrozenConfigDict, ebm: EBM, langevin_gd: bool = None):
    """
    Minimizes the energy for a given state.
    Returns the optimum joint pair of z and a.

    s: (batch_size, state_size)

    returns:
        z: (batch_size, option_size)
        a: (batch_size, action_size)
    """

    if langevin_gd is None:
        langevin_gd = cfg.EBM.LANGEVIN_GD

    # key management
    key, key_init_z, key_init_a, key_infer = jax.random.split(key, 4)

    batch_size = s.shape[0]
    z_init = jax.random.normal(key_init_z, (batch_size, ebm.option_size))
    a_init = jax.random.normal(key_init_a, (batch_size, ebm.action_size))

    z, a = ebm.infer_z_and_a(
        params, s, z_init, a_init, key_infer, langevin_gd,
    )

    return z, a
