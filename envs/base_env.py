from abc import ABC, abstractmethod
import jax
from ml_collections import FrozenConfigDict

from gym import Env as gym_env
from brax.envs.env import Env as brax_env


# TODO: use something that covers brax envs as well
class BaseEnv(ABC):

    def __init__(self, cfg: FrozenConfigDict, key: PRNGKey = None):
        self._prng_key = key
        self._prng_key = key
        if key is None:
            self._prng_key = jax.random.PRNGKey(cfg.seed)


    @property
    def observation_size(self, ):
        pass


    @property
    def action_size(self, ):
        pass
