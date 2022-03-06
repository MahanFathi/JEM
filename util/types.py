from typing import Any
import flax
from jax import numpy as jnp

Params = Any
PRNGKey = jnp.ndarray


@flax.struct.dataclass
class StepData:
    """Subtrajectories are a sequence of StepData entities"""
    observation: jnp.ndarray
    action: jnp.ndarray
