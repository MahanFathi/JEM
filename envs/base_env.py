from gym import Env as gym_env
from brax.envs.env import Env as brax_env


# TODO: use something that covers brax envs as well
class BaseEnv(object):

    def __init__(self, args):
        pass


    @property
    def observation_size(self, ):
        pass


    @property
    def action_size(self, ):
        pass
