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


    def init(self, key: PRNGKey):
        return self._ebm_net.init(key)


    @partial(jax.jit, static_argnums=(0,))
    def apply(self, params: Params, s: jnp.ndarray, z: jnp.ndarray, a: jnp.ndarray):
        return self._ebm_net.apply(params, jnp.concatenate([s, z, a], axis=-1)).squeeze() # (batch_size, 1).squeeze()

