from gym import ActionWrapper
from .transform import *


class FlattenedActionWrapper(ActionWrapper):
    """ Flattens the action space of an `env` using
        `transform.flatten()`. This means that multiple
        discrete actions are joined to a single discrete
        action, and continuous (Box) spaces to a single
        vector valued action.
        The `reverse_action` method is currently not implemented.
    """
    def __init__(self, env):
        super(FlattenedActionWrapper, self).__init__(env)
        trafo = flatten(env.action_space)
        self.action_space = trafo.target
        self.action = trafo.convert_from


class DiscretizedActionWrapper(ActionWrapper):
    """ Discretizes the action space of an `env` using
        `transform.discretize()`.
        The `reverse_action` method is currently not implemented.
    """
    def __init__(self, env, steps):
        super(DiscretizedActionWrapper, self).__init__(env)
        trafo = discretize(env.action_space, steps)
        self.action_space = trafo.target
        self.action = trafo.convert_from


class RescaledActionWrapper(ActionWrapper):
    """ Rescales the action space of an `env` using
        `transform.rescale()`.
        This is useful in case an algorithm is designed to
        produce zero-centered actions (a symmetric action space)
        but the environments actions are non-symmetric.
        The `reverse_action` method is currently not implemented.
    """
    def __init__(self, env, low, high):
        super(RescaledActionWrapper, self).__init__(env)
        trafo = rescale(env.action_space, low=low, high=high)
        self.action_space = trafo.target
        self.action = trafo.convert_from
