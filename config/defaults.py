import ml_collections

# ---------------------------------------------------------------------------- #
# Define Config Node
# ---------------------------------------------------------------------------- #
_C = ml_collections.ConfigDict()
_C.EXP_NAME = ""
_C.seed = 0
_C.WANDB = True
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
_C.SAMPLER.HORIZON = 2
_C.SAMPLER.BATCH_SIZE = 16


# ---------------------------------------------------------------------------- #
# ENERGY-BASED MODEL
# ---------------------------------------------------------------------------- #
_C.EBM = ml_collections.ConfigDict()
_C.EBM.ARCH = "arch0" # {"arch0": simple_feed_forward, "arch1": multipe_mlps}
_C.EBM.ARCH0 = ml_collections.ConfigDict()
_C.EBM.ARCH0.LAYERS = [64, 64, 64, 16]
_C.EBM.ARCH1 = ml_collections.ConfigDict()
_C.EBM.ARCH1.F_LAYERS = [64, 32, 32, 16]
_C.EBM.ARCH1.G_LAYERS = [64, 32, 32, 16]
# _C.EBM.ARCH2.M_LAYERS = [64, 32, 32, 16]
# _C.EBM.ARCH2.E_LAYERS = [64, 32, 32, 16]
_C.EBM.OPTION_TYPE_DISCRETE = False # it's either discrete (1-hot) or continuous
_C.EBM.OPTION_SIZE = 1
_C.EBM.ALPHA = 5e-3 # internal GD step size
_C.EBM.LANGEVIN_GD = True # if True do GD with Langevin noise
_C.EBM.K = 10 # internal optimization #steps
_C.EBM.GRAD_CLIP = 10.0 # grad clipping during inference. 0.0 -> no clipping


# ---------------------------------------------------------------------------- #
# DERIVATIVE-FREE OPTIMIZATION SETTINGS
# ---------------------------------------------------------------------------- #
_C.EBM.DF_OPT = ml_collections.ConfigDict()
_C.EBM.DF_OPT.NUM_ITERATIONS = 128
_C.EBM.DF_OPT.NUM_SAMPLES_PER_DIM = 32
_C.EBM.DF_OPT.STD = 0.4
_C.EBM.DF_OPT.SHRINK_COEFF = 0.95


# ---------------------------------------------------------------------------- #
# EBM TRAIN
# ---------------------------------------------------------------------------- #
_C.TRAIN = ml_collections.ConfigDict()
_C.TRAIN.MAX_DEVICES_PER_HOST = 8
_C.TRAIN.EBM = ml_collections.ConfigDict()
_C.TRAIN.EBM.WARMSTART_INFERENCE = False
_C.TRAIN.EBM.HINGE_EPS = 0.1
_C.TRAIN.EBM.NEGATIVE_ACTION_SAMPLE_SIZE = 64
_C.TRAIN.EBM.ACTION_INFER_BATCH_SIZE = 64
_C.TRAIN.EBM.OPTION_INFER_BATCH_SIZE = 64
_C.TRAIN.EBM.LOSS_NAME = "loss_ML_KL_KLz" # from [loss_ML, loss_ML_KL, loss_L2, loss_L2_KL, loss_contrastive, loss_ML_KL_KLz]
_C.TRAIN.EBM.DATA_SIZE = 1e8 # in case of limited experience
_C.TRAIN.EBM.LEARNING_RATE = 1e-5
_C.TRAIN.EBM.NUM_EPOCHS = 10000
_C.TRAIN.EBM.NUM_UPDATE_EPOCHS = 8
_C.TRAIN.EBM.NUM_SAMPLERS = 8
_C.TRAIN.EBM.BATCH_SIZE = 2 ** 11
_C.TRAIN.EBM.EVAL_BATCH_SIZE = 2 ** 11
_C.TRAIN.EBM.NUM_MINIBATCHES = 8
_C.TRAIN.EBM.NORMALIZE_OBSERVATIONS = False
_C.TRAIN.EBM.NORMALIZE_ACTIONS = False # needs propper support in the code
_C.TRAIN.EBM.LOG_FREQUENCY = 1000
_C.TRAIN.EBM.LOG_SAVE_PARAMS = False
_C.TRAIN.EBM.DISCOUNT = 0.95
_C.TRAIN.EBM.LOSS_KL_COEFF = 1.0
_C.TRAIN.EBM.LOSS_KLZ_COEFF = 1.0

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
