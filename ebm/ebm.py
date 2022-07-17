from functools import partial

import jax
from jax import numpy as jnp
import jaxopt
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
        self._dedz = jax.jit(jax.vmap(jax.grad(self.apply, 2),
                                      in_axes=(None, 0, 0, 0)))
        self._deda = jax.jit(jax.vmap(jax.grad(self.apply, 3),
                                      in_axes=(None, 0, 0, 0)))

        if cfg.EBM.GRAD_CLIP:
            self.dedz = lambda params, s, z, a: jnp.clip(
                self._dedz(params, s, z, a),
                -cfg.EBM.GRAD_CLIP, cfg.EBM.GRAD_CLIP)
            self.deda = lambda params, s, z, a: jnp.clip(
                self._deda(params, s, z, a),
                -cfg.EBM.GRAD_CLIP, cfg.EBM.GRAD_CLIP)
        else:
            self.dedz = self._dedz
            self.deda = self._deda

        # pick inner optimizer
        self.infer_z = self._infer_z_jaxopt if cfg.EBM.JAXOPT.JAXOPT else self._infer_z
        self.infer_a = self._infer_a_jaxopt if cfg.EBM.JAXOPT.JAXOPT else self._infer_a

        # get the energies for fixed (s, z) and a batch of actions
        self.apply_batch_a = jax.vmap(self.apply,
                                      in_axes=(None, None, None, 0))
        # get the energies for fixed (s, a) and a batch of options
        self.apply_batch_z = jax.vmap(self.apply,
                                      in_axes=(None, None, 0, None))
        # langevin batch inferece for actions
        self.infer_batch_a = jax.vmap(self.infer_a,
                                      in_axes=(None, None, None, 0, 0, None))
        # langevin batch inferece for options
        self.infer_batch_z = jax.vmap(self.infer_z,
                                      in_axes=(None, None, 0, None, 0, None))
        # outputs inferred actions for a batch of (s, z)
        self.infer_batch_a_derivative_free = jax.vmap(self.infer_a_derivative_free,
                                      in_axes=(None, 0, 0, 0))


    def init(self, key: PRNGKey):
        return self._ebm_net.init(key)


    @partial(jax.jit, static_argnums=(0,))
    def apply(self, params: Params, s: jnp.ndarray, z: jnp.ndarray, a: jnp.ndarray):
        return self._ebm_net.apply(params, s, z, a).squeeze(axis=-1) ** 2 # (batch_size, 1).squeeze()


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


    def _infer_z(self, params: Params, s: jnp.ndarray, z: jnp.ndarray, a: jnp.ndarray, key: PRNGKey, langevin_gd: bool = None):

        if langevin_gd is None:
            langevin_gd = self.cfg.EBM.LANGEVIN_GD

        (_, _, z, _, _, _), _ = jax.lax.scan(
            self._step_z_grad_descent,
            (params, s, z, a, key, langevin_gd), (), self.cfg.EBM.K)

        return z


    def _infer_a(self, params: Params, s: jnp.ndarray, z: jnp.ndarray, a: jnp.ndarray, key: PRNGKey, langevin_gd: bool = None):

        if langevin_gd is None:
            langevin_gd = self.cfg.EBM.LANGEVIN_GD

        (_, _, _, a, _, _), _ = jax.lax.scan(
            self._step_a_grad_descent,
            (params, s, z, a, key, langevin_gd), (), self.cfg.EBM.K)

        return a


    def _infer_z_jaxopt(self, params: Params, s: jnp.ndarray, z: jnp.ndarray, a: jnp.ndarray, key: PRNGKey, langevin_gd: bool = None):
        optimizer = getattr(jaxopt, self.cfg.EBM.JAXOPT.OPTIMIZER)
        solver = optimizer(
            fun=lambda z, params_ebm, s, a: self.apply(params_ebm, s, z, a),
            maxiter=self.cfg.EBM.JAXOPT.MAXITER, implicit_diff=self.cfg.EBM.JAXOPT.IMP_DIFF,
        )
        return solver.run(z, params_ebm=params, s=s, a=a).params


    def _infer_a_jaxopt(self, params: Params, s: jnp.ndarray, z: jnp.ndarray, a: jnp.ndarray, key: PRNGKey, langevin_gd: bool = None):
        optimizer = getattr(jaxopt, self.cfg.EBM.JAXOPT.OPTIMIZER)
        solver = optimizer(
            fun=lambda a, params_ebm, s, z: self.apply(params_ebm, s, z, a),
            maxiter=self.cfg.EBM.JAXOPT.MAXITER, implicit_diff= self.cfg.EBM.JAXOPT.IMP_DIFF,
        )
        return solver.run(a, params_ebm=params, s=s, z=z).params


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


    def scan_to_infer_multiple_batch_a(self, carry, x: StepData):
        params, z, key, langevin_gd = carry
        s = x.observation
        a_init = x.action

        a_inference_batch_size = a_init.shape[0]

        key, key_infer_a = jax.random.split(key)
        key_infer_a = jax.random.split(key_infer_a, a_inference_batch_size)

        a = self.infer_batch_a(params, s, z, a_init, key_infer_a, langevin_gd)

        return (params, z, key, langevin_gd), a


    def scan_to_iter_dfo(self, carry, unused_t):
        params: Params
        s: jnp.ndarray # single
        z: jnp.ndarray # single
        a: jnp.ndarray # batch
        key: PRNGKey
        params, s, z, a, key, std, energies = carry

        # probs = jax.nn.softmax(-energies, axis=0) # sum across `sample_batch_size`
        # probs size: (sample_batch_size, batch_size)

        sample_batch_size, batch_size, action_size = a.shape
        key, key_sample = jax.random.split(key)
        # update a
        indices = jax.random.categorical(key, -energies, axis=0, shape=(sample_batch_size, batch_size, ))
        pseudo_indices = jnp.stack(sample_batch_size * [jnp.arange(batch_size)])
        a = a[indices, pseudo_indices, :]
        key, key_noise = jax.random.split(key)
        a += jax.random.normal(key_noise, (sample_batch_size, batch_size, action_size)) * std
        a = jnp.clip(a, a_min=0., a_max=1.)
        std *= self.cfg.EBM.DF_OPT.SHRINK_COEFF

        energies = self.apply_batch_a(params, s, z, a)
        # energies size: (sample_batch_size, batch_size)

        return (params, s, z, a, key, std, energies), ()


    def infer_a_derivative_free(self, params: Params, s: jnp.ndarray, z: jnp.ndarray, key: PRNGKey):
        """
        Infers the actions for a batch of observations and options, w/o langevin dynamics.

        s: (batch_size, observation_size)
        z: (batch_size, option_size)

        returns:
            a: (batch_size, action_size)
        """
        action_size = self.env.action_size
        assert s.ndim > 1
        batch_size = s.shape[0]
        sample_batch_size = self.cfg.EBM.DF_OPT.NUM_SAMPLES_PER_DIM ** action_size

        # assuming all actions lie in range [0, 1]
        # init to random values
        key, key_init_a = jax.random.split(key)
        a = jax.random.uniform(key_init_a, (sample_batch_size, batch_size, action_size))
        energies = self.apply_batch_a(params, s, z, a) # size: (sample_batch_size, batch_size, )

        (_, _, _, a, _, _, energies), _ = jax.lax.scan(
            self.scan_to_iter_dfo,
            (params, s, z, a, key, self.cfg.EBM.DF_OPT.STD, energies), (),
            self.cfg.EBM.DF_OPT.NUM_ITERATIONS,
        )
        return a[jnp.argmin(energies, axis=0), jnp.arange(batch_size), :]


# ---------------------------------------------------------------------------- #
# Util Functions
# ---------------------------------------------------------------------------- #
def infer_z(params: Params, data: StepData, key: PRNGKey, cfg: FrozenConfigDict, ebm: EBM, langevin_gd: bool = None):
    """
    Infers the option from the first transition in data.

    data: (batch_size, horizon, dim)

    returns:
        z: (batch_size, option_size)
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
    return z


def infer_batch_z(params: Params, data: StepData, key: PRNGKey, cfg: FrozenConfigDict, ebm: EBM, langevin_gd: bool = None):
    """
    Infers a batch of options from different initializations from the first transition in data.

    data: (batch_size, horizon, dim)

    returns:
        z: (option_infer_batch_size, batch_size, option_size)
    """
    if langevin_gd is None:
        langevin_gd = cfg.EBM.LANGEVIN_GD

    batch_size, horizon, _ = data.action.shape
    option_size = cfg.EBM.OPTION_SIZE
    option_infer_batch_size = cfg.TRAIN.EBM.OPTION_INFER_BATCH_SIZE

    # infer z from first state-action
    key, key_init_z, key_infer_z = jax.random.split(key, 3)
    z_init = jax.random.normal(key_init_z, (option_infer_batch_size, batch_size, option_size))
    key_infer_z = jax.random.split(key_infer_z, option_infer_batch_size)
    z = ebm.infer_batch_z(
        params, data.observation[:, 0, :], z_init,
        data.action[:, 0, :], key_infer_z, langevin_gd,
    )
    return z


def infer_z_then_a(params: Params, data: StepData, key: PRNGKey, cfg: FrozenConfigDict, ebm: EBM, langevin_gd: bool = None):
    """
    Infers the option from the first transition in the data and
    then predicts future actions based on the inferred option.

    This function makes various (#cfg.TRAIN.EBM.ACTION_INFER_BATCH_SIZE)
    future action predictions, which is required by loss KL.

    data: (batch_size, horizon, dim)

    returns:
        z: (batch_size, option_size)
        a: (action_infer_batch_size, batch_size, horizon - 1, action_size)
    """

    # infer z from first state-action
    key, key_infer_z = jax.random.split(key)
    z = infer_z(params, data, key_infer_z, cfg, ebm, langevin_gd)

    if langevin_gd is None:
        langevin_gd = cfg.EBM.LANGEVIN_GD

    batch_size, horizon, action_size = data.action.shape

    # infer actions based on the inferred option
    action_infer_batch_size = cfg.TRAIN.EBM.ACTION_INFER_BATCH_SIZE
    key, key_init_a, key_infer_a = jax.random.split(key, 3)
    if cfg.TRAIN.EBM.WARMSTART_INFERENCE:
        a0 = data.action[:, 0, :]
        a_init = jnp.stack((horizon - 1) * [jnp.stack(action_infer_batch_size * [a0])])
        # TODO: warmstart next inference based on last inferred action
    else:
        a_init = jax.random.uniform( # TODO: assuming all action lie in range [0, 1]
            key_init_a,
            (horizon - 1, action_infer_batch_size, batch_size, action_size),
            minval=-1.,
            maxval=1.,
        )
    _, a = jax.lax.scan(
        ebm.scan_to_infer_multiple_batch_a, (params, z, key_infer_a, langevin_gd),
        StepData(
            observation=data.observation.swapaxes(0, 1)[1:],
            action=a_init,
        ),
    )

    a = a.transpose([1, 2, 0, 3])

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
