from abc import ABC, abstractmethod
from typing import Union

import jax
from ml_collections import FrozenConfigDict

from envs.base_env import BaseEnv
from util.types import *


class BaseSampler(ABC):


    def __init__(self, cfg: FrozenConfigDict, env: BaseEnv, seed: int = 0):
        self.cfg = cfg
        self.horizon = cfg.SAMPLER.HORIZON
        self.batch_size = cfg.SAMPLER.BATCH_SIZE

        self.env = env

        self._prng_key = jax.random.PRNGKey(seed)


    @abstractmethod
    def sample_subtrajectory(self, key: PRNGKey = None):
        pass


    @abstractmethod
    def sample_batch_subtrajectory(self, key: PRNGKey = None):
        pass
