# classify spaces
import gym
from gym import spaces

def assert_space(space):
    """ Raise a `TypeError` exception if `space` is not a `gym.spaces.Space`. """
    if not isinstance(space, gym.Space):
        raise TypeError("Expected a gym.spaces.Space, got {}".format(type(space)))

def is_discrete(space):
    """ Checks if a space is discrete. A space is considered to
        be discrete if it is derived from Discrete, MultiDiscrete
        or MultiBinary.
        A Tuple space is discrete if it contains only discrete 
        subspaces.
        If `space` is an instance of another type, `TypeError` is
        raised.
    """
    assert_space(space)

    if isinstance(space, (spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary)):
        return True
    elif isinstance(space, spaces.Box):
        return False
    elif isinstance(space, spaces.Tuple):
        return all(map(is_discrete, space.spaces))
    raise TypeError("Unknown space {} supplied".format(type(space)))

def is_compound(space):
    """ Checks whether a space is a compound space. These are non-scalar
        `Box` spaces, `MultiDiscrete`, `MultiBinary` and `Tuple` spaces
        (A Tuple space with a single, non-compound subspace is still considered
        compound).
        Any other type of space raises a `TypeError`.
    """
    assert_space(space)

    if isinstance(space, spaces.Discrete):
        return False
    elif isinstance(space, spaces.Box):
        return len(space.shape) != 1 or space.shape[0] != 1
    elif isinstance(space, (spaces.MultiDiscrete, spaces.MultiBinary)):
        return True
    elif isinstance(space, spaces.Tuple):
        return True

    raise TypeError("Unknown space {} supplied".format(type(space)))

