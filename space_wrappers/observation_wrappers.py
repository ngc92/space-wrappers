from gym import ObservationWrapper
from .transform import *


class FlattenedObservationWrapper(ObservationWrapper):
    """
    Wraps the env such that the new env has a flattened
    observation space.
    """
    def __init__(self, env):
        super(FlattenedObservationWrapper, self).__init__(env)
        trafo = flatten(env.observation_space)
        self.observation_space = trafo.target
        self.observation = trafo.convert_to


class DiscretizedObservationWrapper(ObservationWrapper):
    """
    Wraps the env such that the new env has a discrete
    observation space.
    """
    def __init__(self, env, steps):
        super(DiscretizedObservationWrapper, self).__init__(env)
        trafo = discretize(env.observation_space, steps)
        self.observation_space = trafo.target
        self.observation = trafo.convert_to


class RescaledObservationWrapper(ObservationWrapper):
    """
    Wraps the env such that the new env has a rescaled
    observation space.
    """
    def __init__(self, env, low, high):
        super(RescaledObservationWrapper, self).__init__(env)
        trafo = rescale(env.observation_space, low=low, high=high)
        self.observation_space = trafo.target
        self.observation = trafo.convert_to
