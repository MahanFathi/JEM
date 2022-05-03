import jax
from ml_collections import FrozenConfigDict
from util.types import *
from .envs.binary import BinaryEnv, BinarySampler
from .envs.particle_and_target import ParticleAndTargetEnv, ParticleAndTargetSampler

__all__ = []
__all__ += ["BinaryEnv", "BinarySampler"]
__all__ += ["ParticleAndTargetEnv", "ParticleAndTargetSampler"]

_registry = {
    "env": {
        "binary": BinaryEnv,
        "particle_and_target": ParticleAndTargetEnv,
    },
    "sampler": {
        "binary": BinarySampler,
        "particle_and_target": ParticleAndTargetSampler,
    },
}

def build_env_sampler(cfg: FrozenConfigDict, key: PRNGKey = None):
    if key is None:
        key_env, key_sampler = None, None
    else:
        key_env, key_sampler = jax.random.split(key, 2)
    env = _registry["env"][cfg.ENV.ENV_NAME](cfg, key_env)
    sampler = _registry["sampler"][cfg.SAMPLER.SAMPLER_NAME](cfg, env, key_sampler)
    return env, sampler
