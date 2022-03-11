from typing import Union
from functools import partial
import collections
from ml_collections import FrozenConfigDict
import numpy as np
import jax
from jax import numpy as jnp
from brax.envs.env import Env as brax_env
from gym import spaces

from envs.base_env import BaseEnv
from envs.base_sampler import BaseSampler
from util.types import *


class ParticleAndTargetEnv(BaseEnv):


    def __init__(self, cfg: FrozenConfigDict, key: PRNGKey = None):

        super(ParticleAndTargetEnv, self).__init__(cfg: FrozenConfigDict, key: PRNGKey = None)

        self.n_dim = 2 # for now assume 2D
        self.observation_space = self._create_observation_space()
        self._observation_size = 2 * self.n_dim
        self._action_size = self.n_dim
        self.action_space = spaces.Box(low=0., high=1., shape=(self.n_dim,), dtype=np.float32)
        self.particle_step_size = 0.05 # actions are the displacement of the particle
        self.reset()


    @property
    def observation_size(self, ):
        return self._observation_size


    @property
    def action_size(self, ):
        return self._action_size


    def _create_observation_space(self, ):
        obs_dict = collections.OrderedDict(
            pos_agent=spaces.Box(low=0., high=1., shape=(self.n_dim,), dtype=np.float32),
            pos_target=spaces.Box(low=0., high=1., shape=(self.n_dim,), dtype=np.float32),
        )
        return spaces.Dict(obs_dict)


    def reset(self, key: PRNGKey = None):
        if key is None:
            self._prng_key, key = jax.random.split(self._prng_key)
        state = jax.random.uniform(key, (4, ))
        return state


    @partial(jax.jit, static_argnums=(0, ))
    def step(self, state: jnp.ndarray, action: jnp.ndarray):
        angle = action[0] * 2 * jnp.pi
        step_size = action[1] * self.particle_step_size
        pos_agent, pos_target = jnp.split(state, 2)
        pos_agent += jnp.array([jnp.cos(angle), jnp.sin(angle)]) * step_size
        state = jnp.concatenate([pos_agent, pos_target])
        done = False
        reward = self._get_reward()
        return state, reward, done, {}


    def _get_reward(self, ):
        return 0 # we don't care about the reward for now


# expert policies
@jax.jit
def move_towards_target(state: jnp.ndarray):
    pos_agent, pos_target = jnp.split(state, 2)
    dir = pos_target - pos_agent
    angle = jnp.arctan2(*dir[::-1])
    angle += 2 * jnp.pi
    angle = jnp.mod(angle, 2 * jnp.pi)
    return jnp.array([angle / (2 * jnp.pi), 1.])


@jax.jit
def move_away_from_target(state: jnp.ndarray):
    pos_agent, pos_target = jnp.split(state, 2)
    dir = pos_agent - pos_target
    angle = jnp.arctan2(*dir[::-1])
    angle += 2 * jnp.pi
    angle = jnp.mod(angle, 2 * jnp.pi)
    return jnp.array([angle / (2 * jnp.pi), 1.])


@jax.jit
def circle_target_clockwise(state: jnp.ndarray):
    pos_agent, pos_target = jnp.split(state, 2)
    dir = pos_target - pos_agent
    angle = jnp.arctan2(*dir[::-1])
    angle += 2 * jnp.pi
    angle -= jnp.pi / 2.
    angle = jnp.mod(angle, 2 * jnp.pi)
    return jnp.array([angle / (2 * jnp.pi), 1.])


@jax.jit
def circle_target_counterclockwise(state: jnp.ndarray):
    pos_agent, pos_target = jnp.split(state, 2)
    dir = pos_target - pos_agent
    angle = jnp.arctan2(*dir[::-1])
    angle += 2 * jnp.pi
    angle += jnp.pi / 2.
    angle = jnp.mod(angle, 2 * jnp.pi)
    return jnp.array([angle / (2 * jnp.pi), 1.])


# sampler
class ParticleAndTargetSampler(BaseSampler):


    def __init__(self, cfg: FrozenConfigDict, env: BaseEnv, key: PRNGKey = None):
        super(ParticleAndTargetSampler, self).__init__(cfg, env, seed)
        self._policies = [
                    move_towards_target,
                    move_away_from_target,
                    circle_target_clockwise,
                    circle_target_counterclockwise,
                ]
        self._sample_step_fns = [self._make_step_fn(p) for p in self._policies]
        self._batched_reset = jax.vmap(self.env.reset)
        self._sample_batch_subtrajectory = jax.vmap(self._sample_subtrajectory, in_axes=(None, 0, None))
        assert self.batch_size % len(self._policies) == 0, "cfg.SAMPLER.BATCH_SIZE not divisable by len policies."


    def _make_step_fn(self, policy):

        @jax.jit
        def sample_step(carry, unused_t):
            key, state = carry
            action = policy(state)
            new_state, _, _, _ = self.env.step(state, action)
            return (key, new_state), StepData(
                observation=state,
                action=action,
            )

        return sample_step


    def _sample_subtrajectory(self, key: PRNGKey, state, sample_step_fn):
        _, subtrajectory = jax.lax.scan(
            sample_step_fn, (key, state), (), self.horizon)
        return subtrajectory


    def sample_subtrajectory(self, key: PRNGKey = None):
        # toss a coin and choose a policy
        if key is None:
            self._prng_key, key = jax.random.split(self._prng_key)
        policy_id = jax.random.randint(key, (), 0, len(self._policies) - 1)
        state = self.env.reset(key)

        return self._sample_subtrajectory(key, state, self._sample_step_fns[policy_id])


    def sample_batch_subtrajectory(self, key: PRNGKey = None):
        if key is None:
            self._prng_key, key = jax.random.split(self._prng_key)
        key_sample, key_states = jax.random.split(key, 2)
        keys_states = jax.random.split(key_states, self.batch_size)
        states = self._batched_reset(keys_states)
        batch_size_per_policy = self.batch_size // len(self._policies)
        states_splits = jnp.split(states, len(self._policies))
        keys_sample = jax.random.split(key_sample, batch_size_per_policy)
        batch_subtrajectories = []
        for i in range(len(self._policies)):
            key = keys_sample[i]
            sample_step_fn = self._sample_step_fns[i]
            states = states_splits[i]
            batch_subtrajectories.append(self._sample_batch_subtrajectory(key, states, sample_step_fn))

        return jax.tree_multimap(lambda *args: jnp.concatenate(args), *batch_subtrajectories)
