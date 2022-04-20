import ml_collections

# ---------------------------------------------------------------------------- #
# Define Config Node
# ---------------------------------------------------------------------------- #
_C = ml_collections.ConfigDict()
_C.seed = 0
_C.DEBUG = False
_C.MOCK_TPU = False


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
_C.SAMPLER.SAMPLER_NAME = "particle_and_target"
_C.SAMPLER.HORIZON = 5
_C.SAMPLER.BATCH_SIZE = 16


# ---------------------------------------------------------------------------- #
# ENERGY-BASED MODEL
# ---------------------------------------------------------------------------- #
_C.EBM = ml_collections.ConfigDict()
_C.EBM.LAYERS = [64, 32, 32, 16]
_C.EBM.OPTION_TYPE_DISCRETE = False # it's either discrete (1-hot) or continuous
_C.EBM.OPTION_SIZE = 1
_C.EBM.ALPHA = 1e-3 # internal GD step size
_C.EBM.LANGEVIN_GD = False # if True do GD with Langevin noise
_C.EBM.K = 10 # internal optimization #steps


# ---------------------------------------------------------------------------- #
# EBM TRAIN
# ---------------------------------------------------------------------------- #
_C.TRAIN = ml_collections.ConfigDict()
_C.TRAIN.MAX_DEVICES_PER_HOST = 8
_C.TRAIN.EBM = ml_collections.ConfigDict()
_C.TRAIN.EBM.LOSS_NAME = "loss_ML_KL" # from [loss_ML, loss_ML_KL, loss_L2, loss_L2_KL]
_C.TRAIN.EBM.DATA_SIZE = 1e6
_C.TRAIN.EBM.LEARNING_RATE = 1e-6
_C.TRAIN.EBM.NUM_EPOCHS = 1000
_C.TRAIN.EBM.NUM_UPDATE_EPOCHS = 8
_C.TRAIN.EBM.NUM_SAMPLERS = 8
_C.TRAIN.EBM.BATCH_SIZE = 2 ** 13
_C.TRAIN.EBM.EVAL_BATCH_SIZE = 2 ** 12
_C.TRAIN.EBM.NUM_MINIBATCHES = 8
_C.TRAIN.EBM.NORMALIZE_OBSERVATIONS = True
_C.TRAIN.EBM.NORMALIZE_ACTIONS = True
_C.TRAIN.EBM.LOG_FREQUENCY = 20
_C.TRAIN.EBM.DISCOUNT = 0.9
_C.TRAIN.EBM.LOSS_KL_COEFF = 1.0

# ---------------------------------------------------------------------------- #
# batch guide:
# ---------------------------------------------------------------------------- #
#   gradient are calculated based on batches of size:
#       `TRAIN.EBM.BATCH_SIZE // TRAIN.EBM.NUM_MINIBATCHES`.
#   this has been made possible by pmapping batches of size:
#       `TRAIN.EBM.BATCH_SIZE // #local_devices // TRAIN.EBM.NUM_MINIBATCHES`,
#   across `#local_devices` local devices per CPU host/node, and
#   calculating the mean of grad via `jax.lax.pmean`.
#
#   This process is repeated for `TRAIN.EBM.NUM_UPDATE_EPOCHS`
#   times per epoch and we run `TRAIN.EBM.NUM_EPOCHS` epochs
#   in total.
# ---------------------------------------------------------------------------- #

def get_config():
    return _C
