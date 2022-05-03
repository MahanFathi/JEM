from typing import Any, Callable, Sequence, Tuple

import dataclasses
from flax import linen
import jax
import jax.numpy as jnp
from ml_collections import FrozenConfigDict


@dataclasses.dataclass
class Model:
    init: Any
    apply: Any


class MLP(linen.Module):
    """MLP module."""
    layer_sizes: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.elu
    kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = linen.Dense(
                hidden_size,
                name=f'hidden_{i}',
                kernel_init=self.kernel_init,
                use_bias=self.bias)(hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                hidden = self.activation(hidden)
        return hidden


def make_mlp(layer_sizes: Sequence[int],
             input_size: int,
             activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.swish,
             ) -> Model:
    """Creates a model.

    Args:
      layer_sizes: layers
      input_size: size of an observation
      activation: activation

    Returns:
      a model
    """
    dummy_obs = jnp.zeros((1, input_size))
    module = MLP(layer_sizes=layer_sizes, activation=activation)

    model = Model(
        init=lambda rng: module.init(rng, dummy_obs), apply=module.apply)
    return model


def make_ebm_model_arch0(cfg: FrozenConfigDict, observation_size: int, action_size: int):

    def make_apply_fn(module):
        return lambda params, s, z, a: module.apply(params, jnp.concatenate([s, z, a], axis=-1))

    ebm_net = make_mlp(
        list(cfg.EBM.ARCH0.LAYERS) + [1],
        observation_size + cfg.EBM.OPTION_SIZE + action_size,
    )

    return Model(
        init=ebm_net.init,
        apply=make_apply_fn(ebm_net),
    )


def make_ebm_model_arch1(cfg: FrozenConfigDict, observation_size: int, action_size: int):

    @dataclasses.dataclass
    class Arch1Params:
        f: Any
        g: Any

    def make_apply_fn(f, g):
        def apply_fn(params: Arch1Params, s, z, a):
            y = f.apply(params['f'], jnp.concatenate([s, a], axis=-1))
            e = g.apply(params['g'], jnp.concatenate([z, y], axis=-1))
            return e
        return apply_fn

    def make_init_fn(f, g):
        def init_fn(rng):
            key_f, key_g = jax.random.split(rng)
            arch1params = Arch1Params(
                f=f.init(key_f),
                g=g.init(key_g),
            )
            # NOTE: params need to be a dict by optax
            return dataclasses.asdict(arch1params)
        return init_fn

    f = make_mlp(
        list(cfg.EBM.ARCH1.F_LAYERS),
        observation_size + action_size,
    )
    g = make_mlp(
        list(cfg.EBM.ARCH1.G_LAYERS) + [1],
        cfg.EBM.OPTION_SIZE + list(cfg.EBM.ARCH1.F_LAYERS)[-1],
    )

    return Model(
        init=make_init_fn(f, g),
        apply=make_apply_fn(f, g),
    )
