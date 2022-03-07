from functools import partial

import jax
from jax import numpy as jnp
from ml_collections import FrozenConfigDict

from envs.base_env import BaseEnv
from util.net import make_model
from util.types import *


class EBM(object):


    def __init__(self, cfg: FrozenConfigDict, env: BaseEnv, seed: int = 0):

        self.cfg = cfg
        self.env = env
        self._ebm_net = make_model(
            list(cfg.EBM.LAYERS) + [1],
            env.observation_size + cfg.EBM.OPTION_SIZE + env.action_size,
        )

        # define derivatives
        self.dedz = jax.jit(jax.vmap(jax.grad(self.apply, 2), in_axes=(None, 0, 0, 0)))
        self.deda = jax.jit(jax.vmap(jax.grad(self.apply, 3), in_axes=(None, 0, 0, 0)))

        self._seed(seed)


    def _seed(self, seed):
        self._prng_key = jax.random.PRNGKey(seed)


    def init(self, key: PRNGKey):
        return self._ebm_net.init(key)


    @partial(jax.jit, static_argnums=(0,))
    def apply(self, params: Params, s: jnp.ndarray, z: jnp.ndarray, a: jnp.ndarray):
        return self._ebm_net.apply(params, jnp.concatenate([s, z, a], axis=-1)).squeeze() # (batch_size, 1).squeeze()


    @partial(jax.jit, static_argnums=(0, ))
    def _step_z_grad_descent(self, carry, unused_t):
        params: Params
        s: jnp.ndarray
        z: jnp.ndarray
        a: jnp.ndarray
        key: PRNGKey
        params, s, z, a, key = carry

        alpha = self.cfg.EBM.ALPHA
        omega = .0
        if self.cfg.EBM.LANGEVIN_GD:
            key, langevin_key = jax.random.split(key)
            omega = jax.random.normal(langevin_key, z.shape) * jnp.sqrt(alpha)

        z += -alpha / 2. * self.dedz(params, s, z, a) + omega

        return (params, s, z, a, key), ()


    @partial(jax.jit, static_argnums=(0, ))
    def _step_a_grad_descent(self, carry, unused_t):
        params: Params
        s: jnp.ndarray
        z: jnp.ndarray
        a: jnp.ndarray
        key: PRNGKey
        params, s, z, a, key = carry

        alpha = self.cfg.EBM.ALPHA
        omega = .0
        if self.cfg.EBM.LANGEVIN_GD:
            key, langevin_key = jax.random.split(key)
            omega = jax.random.normal(langevin_key, a.shape) * jnp.sqrt(alpha)

        a += -alpha / 2. * self.deda(params, s, z, a) + omega

        return (params, s, z, a, key), ()


    def infer_z(self, params: Params, s: jnp.ndarray, z: jnp.ndarray, a: jnp.ndarray, key: PRNGKey = None):

        if key is None:
            self._prng_key, key = jax.random.split(self._prng_key)

        (_, _, _, z, _), _ = jax.lax.scan(
            self._step_z_grad_descent,
            (params, s, z, a, key), (), self.cfg.EBM.K)

        return z


    def infer_a(self, params: Params, s: jnp.ndarray, z: jnp.ndarray, a: jnp.ndarray, key: PRNGKey = None):

        if key is None:
            self._prng_key, key = jax.random.split(self._prng_key)

        (_, _, _, _, a), _ = jax.lax.scan(
            self._step_a_grad_descent,
            (params, s, z, a, key), (), self.cfg.EBM.K)

        return a
