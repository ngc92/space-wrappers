import numpy as np
from gym import spaces


class CMPMultiBinary(spaces.MultiBinary):
    def __eq__(self, other):
        return isinstance(other, spaces.MultiBinary) and other.n == self.n

    def __repr__(self):
        return "MultiBinary(%i)" % self.n


class CMPMultiDiscrete(spaces.MultiDiscrete):
    def __eq__(self, other):
        return isinstance(other, spaces.MultiDiscrete) and np.all(other.nvec == self.nvec)

    def __repr__(self):
        return "MultiDiscrete(%i)" % self.nvec


def expects(space):
    if isinstance(space, (spaces.Discrete, spaces.Box)):
        return space

    if isinstance(space, spaces.MultiBinary):
        return CMPMultiBinary(space.n)

    if isinstance(space, spaces.MultiDiscrete):
        return CMPMultiDiscrete(space.nvec)

