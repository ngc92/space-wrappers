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


def num_discrete_actions(space):
    """
    For a discrete space, gets the number of available actions as a tuple.
    :param gym.Space space: The discrete space which to inspect.
    :return tuple: Tuple of integers containing the number of discrete actions.
    :raises TypeError: If the space is not discrete.
    """
    if not is_discrete(space):
        raise TypeError("Space {} is not discrete".format(space))
    if isinstance(space, spaces.Discrete):
        return tuple((space.n,))
    elif isinstance(space, spaces.MultiDiscrete):
        # add +1 here as space.high is an inclusive bound
        return tuple(space.high - space.low + 1)
    elif isinstance(space, spaces.MultiBinary):
        return (2,) * space.n

    raise NotImplementedError()

