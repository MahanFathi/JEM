from absl import app
from absl import flags

import jax
import ml_collections
from ml_collections.config_flags import config_flags

from envs import build_env_sampler
from train_ebm import train_ebm
from config.defaults import get_config


_CONFIG = config_flags.DEFINE_config_file('cfg', './config/defaults.py')


def main(argv):
    del argv

    # freeze the config
    cfg = ml_collections.FrozenConfigDict(_CONFIG.value)

    key = jax.random.PRNGKey(cfg.seed)
    key_train, key_env = jax.random.split(key)

    env, sampler = build_env_sampler(cfg, key_env)
    train_ebm(cfg, env, sampler, key_train)

if __name__ == '__main__':
    app.run(main)
