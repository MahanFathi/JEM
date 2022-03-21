from typing import Any
import flax
from jax import numpy as jnp

Params = Any
PRNGKey = jnp.ndarray


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    optimizer_state: optax.OptState
    params: Params
    key: PRNGKey
    obs_normalizer_params: Params
    act_normalizer_params: Params



@flax.struct.dataclass
class StepData:
    """Subtrajectories are a sequence of StepData entities"""
    observation: jnp.ndarray
    action: jnp.ndarray
