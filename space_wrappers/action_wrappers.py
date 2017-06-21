from gym import ActionWrapper
from .transform import *

class FlattenedActionWrapper(ActionWrapper):
    def __init__(self, env):
        super(FlattenedActionWrapper, self).__init__(env)
        trafo = flatten(env.action_space)
        self.action_space = trafo.target
        self._action = trafo.convert_from

class DiscretizedActionWrapper(ActionWrapper):
    def __init__(self, env, steps):
        super(DiscretizedActionWrapper, self).__init__(env)
        trafo = discretize(env.action_space, steps)
        self.action_space = trafo.target
        self._action = trafo.convert_from

class RescaledActionWrapper(ActionWrapper):
    def __init__(self, env, low, high):
        super(RescaledActionWrapper, self).__init__(env)
        trafo = rescale(env.action_space, low=low, high=high)
        self.action_space = trafo.target
        self._action = trafo.convert_from
