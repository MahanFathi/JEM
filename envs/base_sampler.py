from abc import ABC, abstractmethod
from typing import Union

import jax
from ml_collections import FrozenConfigDict

from envs.base_env import BaseEnv
from util.types import *


class BaseSampler(ABC):

    def __init__(self, cfg: FrozenConfigDict, env: BaseEnv, key: PRNGKey = None):
        self.cfg = cfg
        self.horizon = cfg.SAMPLER.HORIZON

        self.env = env

        self._prng_key = key
        if key is None:
            self._prng_key = jax.random.PRNGKey(cfg.seed)


    @abstractmethod
    def sample_subtrajectory(self, key: PRNGKey = None):
        pass


    @abstractmethod
    def sample_batch_subtrajectory(self, batch_size: int, key: PRNGKey = None):
        pass
