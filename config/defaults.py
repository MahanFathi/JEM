import ml_collections

# ---------------------------------------------------------------------------- #
# Define Config Node
# ---------------------------------------------------------------------------- #
_C = ml_collections.ConfigDict()


# ---------------------------------------------------------------------------- #
# ENVIRONMENT
# ---------------------------------------------------------------------------- #
_C.ENV = ml_collections.ConfigDict()
_C.ENV.ENV_NAME = "particle_and_target"
_C.ENV.TIMESTEP = 0.05


# ---------------------------------------------------------------------------- #
# SAMPLER
# ---------------------------------------------------------------------------- #
_C.SAMPLER = ml_collections.ConfigDict()
_C.SAMPLER.HORIZON = 5
_C.SAMPLER.BATCH_SIZE = 16


# ---------------------------------------------------------------------------- #
# ENERGY-BASED MODEL
# ---------------------------------------------------------------------------- #
_C.EBM = ml_collections.ConfigDict()
_C.EBM.LAYERS = [64, 32, 32, 16]
_C.EBM.OPTION_TYPE_DISCRETE = False # it's either discrete (1-hot) or continuous
_C.EBM.OPTION_SIZE = 1



def get_cfg_defaults():
    return ml_collections.FrozenConfigDict(_C)
