import ml_collections
from config.defaults import get_config as get_default_config

def get_config():
    _C = get_default_config()

    _C.DEBUG = True

    # mocks 8 tpu devices on cpu
    _C.MOCK_TPU = False

    _C.ENV.ENV_NAME = "particle_and_target"
    _C.SAMPLER.SAMPLER_NAME = "particle_and_target"

    _C.TRAIN.EBM.BATCH_SIZE = 64

    _C.TRAIN.EBM.NUM_EPOCHS = 20
    _C.TRAIN.EBM.LOG_FREQUENCY = 20

    _C.TRAIN.EBM.LEARNING_RATE = 1e-6

    return _C
