# transform spaces
from gym import Space, spaces
import numpy as np
import itertools
import numbers
from collections import namedtuple
from .classify import *

Transform = namedtuple('Transform', ['original', 'target', 'convert_to', 'convert_from'])


# Discretization 
def discretize(space, steps):
    """
    Creates a discretized version of `space` and returns
    a `Transform` that contains the conversion functions.
    If the space is already discrete, the identity
    is returned. The steps are distributed such that the old
    minimum and maximum value can still be reached in the new
    domain.
    :param gym.Space space: The space to be discretized.
    :param int steps: The number of discrete steps to produce
                  for each continuous dimension. Can be an
                  Integer or a list.
    :raises ValueError: If less than two steps are are supplied.
    :return Transform: A `Transform` to the discretized space.
    """

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

            def back(x):
                return int((x - lo) * (steps-1) / (hi - lo))

            return Transform(original=space, target=discrete_space, convert_from=convert, convert_to=back)
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

            def back(x):
                return np.reshape((x - lo) * (steps-1) / (hi - lo), (len(steps),)).astype(int)

            return Transform(original=space, target=discrete_space, convert_from=convert, convert_to=back)
    raise ValueError()


# Flattening
def flatten(space):
    """
    Flattens a space, which means that for continuous spaces (Box)
    the space is reshaped to be of rank 1, and for multidimensional
    discrete spaces a single discrete action with an increased number
    of possible values is created.
    Please be aware that the latter can be potentially pathological in case
    the input space has many discrete actions, as the number of single discrete
    actions increases exponentially ("curse of dimensionality").
    :param gym.Space space: The space that will be flattened
    :return Transform: A transform object describing the transformation
            to the flattened space.
    :raises TypeError, if `space` is not a `gym.Space`.
            NotImplementedError, if the supplied space is neither `Box` nor
            `MultiDiscrete` or `MultiBinary`, and not recodgnized as
            an already flat space by `is_compound`.
    """
    if not isinstance(space, Space):
        raise TypeError("Expected gym.Space, got {}".format(type(space)))

    # no need to do anything if already flat
    if not is_compound(space):
        return Transform(space, space, lambda x: x, lambda x: x)

    if isinstance(space, spaces.Box):
        shape = space.low.shape
        lo = space.low.flatten()
        hi = space.high.flatten()

        def convert(x):
            return np.reshape(x, shape)

        def back(x):
            return np.reshape(x, lo.shape)

        flat_space = spaces.Box(low=lo, high=hi)
        return Transform(original=space, target=flat_space, convert_from=convert, convert_to=back)

    elif isinstance(space, (spaces.MultiDiscrete, spaces.MultiBinary)):
        if isinstance(space, spaces.MultiDiscrete):
            ranges = [range(space.low[i], space.high[i]+1, 1) for i in range(space.num_discrete_space)]
        elif isinstance(space, spaces.MultiBinary):
            ranges = [range(0, 2) for i in range(space.n)]
        prod   = itertools.product(*ranges)
        lookup = list(prod)
        inverse_lookup = {value: key for (key, value) in enumerate(lookup)}
        flat_space = spaces.Discrete(len(lookup))
        convert = lambda x: lookup[x]
        back    = lambda x: inverse_lookup[x]
        return Transform(original=space, target=flat_space, convert_from=convert, convert_to=back)
    else:
        raise NotImplementedError("Does not know how to flatten {}".format(type(space)))


# rescale a continuous action space
def rescale(space, low, high):
    """ A space transform that changes a continuous
        space to a new one with the specified upper and lower
        bounds by linear transformations. If source and target
        range are infinite, only the offset is corrected. If
        one of the ranges is finite and the other infinite, an
        error is raised.
    :rtype: Transform
    :param gym.Space space: The original space. Needs to be
        continuous.
    :param low: Lower bound of the new space.
    :param high: Upper bound of the new space.
    """
    if is_discrete(space):
        raise TypeError("Cannot rescale discrete space {}".format(space))

    if not isinstance(space, spaces.Box):
        raise NotImplementedError()

    # shortcuts
    lo = space.low
    hi = space.high

    # ensure new low/high values are arrays
    if isinstance(low, numbers.Number):
        low = np.ones_like(space.low) * low
    if isinstance(high, numbers.Number):
        high = np.ones_like(space.high) * high

    offset = np.copy(low)
    rg = hi - lo
    rs = high - low
    if np.isnan(rs).any():
        raise ValueError("Invalid range %s to %s specified" % (low, high))

    # the following code is responsible for correctly setting the scale factor and offset
    # in cases where the limits of the ranges become infinite.
    scale_factor = np.zeros_like(lo)
    for i in range(lo.size):
        if np.isinf(rg[i]) and np.isinf(rs[i]):
            scale_factor[i] = 1.0
        else:
            scale_factor[i] = rg[i] / rs[i]

        if low[i] == -np.inf and lo[i] == -np.inf:
            lo[i] = 0.0
            if high[i] == np.inf and hi[i] == np.inf:
                offset[i] = 0.0
            else:
                offset[i] = high[i] - hi[i]

    if np.isinf(scale_factor).any() or (scale_factor == 0.0).any():
        raise ValueError("Cannot map finite to infinite range [%s to %s] to [%s to %s] " % (lo, hi, low, high))

    def convert(x):
        y = (x - offset) * scale_factor # y is in [0, rg]
        return y + lo

    def back(x):
        return (x - lo) / scale_factor + offset

    scaled_space = spaces.Box(low, high)
    return Transform(original=space, target=scaled_space, convert_from=convert, convert_to=back)
