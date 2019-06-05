import gym
from space_wrappers.transform import discretize, flatten, rescale
from gym.spaces import Box, Discrete, MultiDiscrete, MultiBinary, Tuple
import numpy as np
import itertools
import pytest
import numbers


def check_convert(trafo, target, original):
    if isinstance(original, numbers.Number):
        assert trafo.convert_from(target) == original
    else:
        assert (trafo.convert_from(target) == original).all()

    if isinstance(target, numbers.Number):
        assert trafo.convert_to(original) == target
    else:
        assert (trafo.convert_to(original) == target).all()


# discretize
def test_discretize_1d_box():
    cont = Box(np.array([0.0]), np.array([1.0]), dtype=np.float32)
    trafo = discretize(cont, 10)

    assert trafo.target == Discrete(10)
    assert trafo.convert_from(0) == 0.0
    assert trafo.convert_from(9) == 1.0
    assert trafo.convert_to(0.0) == 0
    assert trafo.convert_to(1.0) == 9


def test_discretize_discrete():
    start = Discrete(5)
    trafo = discretize(start, 10)
    assert trafo.target == start
    check_convert(trafo, 3, 3)


def test_discretize_nd_box():
    from space_wrappers.tests.space_equal import expects

    cont = Box(np.array([0.0, 1.0]), np.array([1.0, 2.0]), dtype=np.float32)
    trafo = discretize(cont, 10)

    assert trafo.target == expects(MultiDiscrete([10, 10]))
    check_convert(trafo, (0, 0), [0.0, 1.0])
    check_convert(trafo, (9, 9), [1.0, 2.0])

    trafo = discretize(cont, (5, 10))

    assert trafo.target == expects(MultiDiscrete([5, 10]))
    check_convert(trafo, (0, 0), [0.0, 1.0])
    check_convert(trafo, (4, 9), [1.0, 2.0])

    # check that it also works with np.ndarray data
    check_convert(trafo, np.array([0, 0]), [0.0, 1.0])


def test_discretize_errors():
    cont = Box(np.array([0.0, 1.0]), np.array([1.0, 2.0]), dtype=np.float32)
    with pytest.raises(TypeError):
        trafo = discretize(5, 5)

    with pytest.raises(ValueError):
        trafo = discretize(cont, 1)

    with pytest.raises(NotImplementedError):
        trafo = discretize(Tuple(spaces=[cont]), 10)

    with pytest.raises(ValueError):
        trafo = discretize(cont, [1, 1])

    with pytest.raises(ValueError):
        trafo = discretize(cont, [5, 5, 5])


# flatten
def test_flatten_single():
    start = Discrete(5)
    trafo = flatten(start)
    assert trafo.target == start
    check_convert(trafo, 4, 4)

    start = Box(np.array([0.0]), np.array([1.0]), dtype=np.float32)
    trafo = flatten(start)
    assert trafo.target == start
    check_convert(trafo, 0.5, 0.5)


def test_flatten_discrete():
    md = MultiDiscrete([3, 4])
    trafo = flatten(md)

    assert trafo.target == Discrete(12)
    # check that we get all actions exactly once
    actions = []
    for (i, j) in itertools.product([0, 1, 2], [0, 1, 2, 3]):
        actions += [(i, j)]
    for i in range(0, 12):
        a = trafo.convert_from(i)
        assert a in actions, (a, actions)
        assert trafo.convert_to(a) == i
        actions = list(filter(lambda x: x != a, list(actions)))
    assert len(actions) == 0

    # same test for binary
    md = MultiBinary(3)
    trafo = flatten(md)

    assert trafo.target == Discrete(2**3)
    # check that we get all actions exactly once
    actions = []
    for (i, j, k) in itertools.product([0, 1], [0, 1], [0, 1]):
        actions += [(i, j, k)]
    for i in range(0, 8):
        a = trafo.convert_from(i)
        assert trafo.convert_to(a) == i
        assert a in actions, (a, actions)
        actions = list(filter(lambda x: x != a, actions))
    assert len(actions) == 0

    # check support for numpy array and list
    assert trafo.convert_to((1, 0, 1)) == trafo.convert_to(np.array([1, 0, 1]))
    assert trafo.convert_to((1, 0, 1)) == trafo.convert_to([1, 0, 1])


def test_flatten_continuous():
    ct = Box(np.zeros((2,2)), np.ones((2, 2)), dtype=np.float32)
    trafo = flatten(ct)

    assert trafo.target == Box(np.zeros(4), np.ones(4), dtype=np.float32)
    check_convert(trafo, [1, 2, 3, 4], [[1, 2], [3, 4]])


def test_flatten_tuple():
    s1 = Box(np.zeros(3), np.ones(3), dtype=np.float32)
    s2 = Box(np.ones(2), np.ones(2)*2, dtype=np.float32)
    trafo = flatten(Tuple((s1, s2)))

    assert trafo.target == Box(np.asarray([0.0, 0, 0, 1, 1]), np.asarray([1.0, 1, 1, 2, 2]), dtype=np.float32)
    assert trafo.convert_to(([0, 1, 0], [1, 2])) == pytest.approx([0, 1, 0, 1, 2],)
    assert trafo.convert_from([0, 1, 0, 1, 2]) == (pytest.approx([0, 1, 0]), pytest.approx([1, 2]))


def test_flatten_tuple_recursive():
    s1 = Box(np.zeros((2, 2)), np.ones((2, 2)), dtype=np.float32)
    s2 = Box(np.ones(2), np.ones(2) * 2, dtype=np.float32)
    trafo = flatten(Tuple((s1, s2)))

    assert trafo.target == Box(np.asarray([0.0, 0, 0, 0, 1, 1]), np.asarray([1.0, 1, 1, 1, 2, 2]), dtype=np.float32)
    assert trafo.convert_to(([[0, 1], [1, 0]], [1, 2])) == pytest.approx([0, 1, 1, 0, 1, 2], )
    assert trafo.convert_from([0, 1, 1, 0, 1, 2]) == (pytest.approx(np.asarray([[0, 1], [1, 0]])), pytest.approx([1, 2]))


def test_flatten_errors():
    class UnknownSpace(gym.Space):
        pass

    with pytest.raises(TypeError):
        flatten(5)

    with pytest.raises(NotImplementedError):
        flatten(UnknownSpace())


# rescale
@pytest.mark.parametrize("space", [Discrete(10), MultiDiscrete([2, 3]), MultiBinary(5)])
def test_rescale_discrete(space):
    # cannot rescale discrete spaces
    with pytest.raises(TypeError):
        rescale(space, 0.0, 1.0)


def test_rescale_tuple():
    with pytest.raises(NotImplementedError):
        rescale(Tuple([Box(0, 1, (1, 1), dtype=np.float32)]), 0.0, 1.0)


def test_rescale_box():
    s = Box(np.array([0.0, 1.0]), np.array([1.0, 2.0]), dtype=np.float32)
    trafo = rescale(s, np.array([1.0, 0.0]), np.array([2.0, 1.0]))

    assert trafo.target == Box(np.array([1.0, 0.0]), np.array([2.0, 1.0]), dtype=np.float32)
    check_convert(trafo, [1.0, 0.0], [0.0, 1.0])
    check_convert(trafo, [2.0, 1.0], [1.0, 2.0])

    # scalar rescale
    s = Box(np.array([0.0, 1.0]), np.array([1.0, 2.0]), dtype=np.float32)
    trafo = rescale(s, 0.0, 1.0)

    assert trafo.target == Box(np.array([0.0, 0.0]), np.array([1.0, 1.0]), dtype=np.float32)


def test_rescale_checks():
    # check that invalid target range causes error
    with pytest.raises(ValueError):
        rescale(Box(np.array([0.0]), np.array([1.0]), dtype=np.float32), np.inf, np.inf)

    # cannot linearly transform infinite to finite range
    with pytest.raises(ValueError):
        s = Box(np.array([-np.inf, 0.0]), np.array([np.inf, np.inf]), dtype=np.float32)
        trafo = rescale(s, np.array([1.0, 1.0]), np.array([3.0, np.inf]))


def test_rescale_inf():
    # positive infinite
    s = Box(np.array([0.0, 0.0]), np.array([1.0, np.inf]), dtype=np.float32)
    trafo = rescale(s, np.array([1.0, 1.0]), np.array([3.0, np.inf]))

    assert trafo.target == Box(np.array([1.0, 1.0]), np.array([3.0, np.inf]), dtype=np.float32)
    check_convert(trafo, [3.0, 1.0], [1.0, 0.0])

    # negative infinite
    s = Box(np.array([-1.0, -np.inf]), np.array([0.0, 0.0]), dtype=np.float32)
    trafo = rescale(s, np.array([1.0, -np.inf]), np.array([3.0, 1.0]))

    assert trafo.target == Box(np.array([1.0, -np.inf]), np.array([3.0, 1.0]), dtype=np.float32)
    check_convert(trafo, [1.0, 1.0], [-1.0, 0.0])

    # two sided
    s = Box(np.array([-1.0, -np.inf]), np.array([0.0, np.inf]), dtype=np.float32)
    trafo = rescale(s, np.array([1.0, -np.inf]), np.array([3.0, np.inf]))

    assert trafo.target == Box(np.array([1.0, -np.inf]), np.array([3.0, np.inf]), dtype=np.float32)
    check_convert(trafo, [1.0, 12.0], [-1.0, 12.0])
