# transform spaces
from gym import spaces
import numpy as np
import itertools
import numbers
from collections import namedtuple
from .classify import *

Transform = namedtuple('Transfom', ['original', 'target', 'convert_to', 'convert_from'])


# Discretization 
def discretize(space, steps):
    # there are two possible ways how we could handle already
    # discrete spaces. 
    #  1) throw an error because (unless
    #     steps is configured to fit) we would try to convert 
    #     an already discrete space to one with a different number
    #     of states.
    #  2) keep the space as is.
    # here, we implement the second. This allows scripts that 
    # train a discrete agent to just apply discretize, only 
    # changing envs that are not already discrete.
    if is_discrete(space):
        return Transform(space, space, lambda x: x, lambda x: x)

    # check that step number is valid and convert steps into a np array
    if not isinstance(steps, numbers.Integral):
        steps = np.array(steps, dtype=int)
        if (steps < 2).any():
            raise ValueError("Need at least two steps to discretize, got {}".format(steps))
    elif steps < 2:
        raise ValueError("Need at least two steps to discretize, got {}".format(steps))

    if isinstance(space, spaces.Box):
        if len(space.shape) == 1 and space.shape[0] == 1:
            discrete_space = spaces.Discrete(steps)
            lo = space.low[0]
            hi = space.high[0]
            def convert(x):
                return lo + (hi - lo) * float(x) / (steps-1)
            return Transform(original=space, target=discrete_space, convert_from=convert, convert_to=None)
        else:
            if isinstance(steps, numbers.Integral):
                steps = np.full(space.low.shape, steps)
            assert steps.shape == space.shape, "supplied steps have invalid shape"
            starts = np.zeros_like(steps)
            # MultiDiscrete is inclusive, thus we need steps-1 as last value
            # currently, MultiDiscrete iterates twice over its input, which is not possible for a zip
            # result in python 3
            discrete_space = spaces.MultiDiscrete(list(zip(starts.flatten(), (steps-1).flatten())))
            lo = space.low.flatten()
            hi = space.high.flatten()
            def convert(x):
                return np.reshape(lo + (hi - lo) * x / (steps-1), space.shape)
            return Transform(original=space, target=discrete_space, convert_from=convert, convert_to=None)
    raise ValueError()

# Flattening
def flatten(space):
    # no need to do anything if already flat
    if not is_compound(space):
        return Transform(space, space, lambda x: x, lambda x: x)

    if isinstance(space, spaces.Box):
        shape = space.low.shape
        lo = space.low.flatten()
        hi = space.high.flatten()
        def convert(x):
            return np.reshape(x, shape)
        flat_space = spaces.Box(low=lo, high=hi)
        return Transform(original=space, target=flat_space, convert_from=convert, convert_to=None)

    elif isinstance(space, (spaces.MultiDiscrete, spaces.MultiBinary)):
        if isinstance(space, spaces.MultiDiscrete):
            ranges = [range(space.low[i], space.high[i]+1, 1) for i in range(space.num_discrete_space)]
        elif isinstance(space, spaces.MultiBinary):
            ranges = [range(0, 2) for i in range(space.n)]
        prod   = itertools.product(*ranges)
        lookup = list(prod)
        flat_space = spaces.Discrete(len(lookup))
        convert = lambda x: lookup[x]
        return Transform(original=space, target=flat_space, convert_from=convert, convert_to=None)

    raise TypeError("Cannot flatten {}".format(type(space)))


# rescale a continuous action space
def rescale(space, low, high):
    if is_discrete(space):
        raise TypeError("Cannot rescale discrete space {}".format(space))

    if not isinstance(space, spaces.Box):
        raise TypeError("Cannot rescale non-Box space {}".format(space))

    lo = space.low
    hi = space.high
    rg = hi - lo
    rs = high - low
    sc = rg / rs
    def convert(x):
        y = (x - low) * sc # y is in [0, rg]
        return y + space.low
    scaled_space = spaces.Box(low, high)
    return Transform(original=space, target=scaled_space, convert_from=convert, convert_to=None)