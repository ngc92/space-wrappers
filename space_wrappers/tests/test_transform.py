import gym
from space_wrappers.transform import discretize, flatten, rescale
from gym.spaces import *
import numpy as np
import itertools

def check_convert(trafo, target, original):
    assert (trafo.convert_from(target) == original).all(), "%s != %s" % (trafo.convert_from(target), original)
    assert (trafo.convert_to(original) == target).all(), "%s != %s" % (trafo.convert_to(original), target)

# discretize
def test_discretize_1d_box():
    cont = Box(np.array([0.0]), np.array([1.0]))
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

def test_discretize_nd_box():
    cont = Box(np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    trafo = discretize(cont, 10)

    assert trafo.target == MultiDiscrete([(0, 9), (0, 9)])
    check_convert(trafo, (0, 0), [0.0, 1.0])
    check_convert(trafo, (9, 9), [1.0, 2.0])

    trafo = discretize(cont, (5, 10))

    assert trafo.target == MultiDiscrete([(0, 4), (0, 9)])
    check_convert(trafo, (0, 0), [0.0, 1.0])
    check_convert(trafo, (4, 9), [1.0, 2.0])

# flatten
def test_flatten_single():
    start = Discrete(5)
    trafo = flatten(start)
    assert trafo.target == start

    start = Box(np.array([0.0]), np.array([1.0]))
    trafo = flatten(start)
    assert trafo.target == start

def test_flatten_discrete():
    md = MultiDiscrete([(0, 2), (0, 3)])
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

def test_flatten_continuous():
    ct = Box(np.zeros((2,2)), np.ones((2, 2)))
    trafo = flatten(ct)

    assert trafo.target == Box(np.zeros(4), np.ones(4))
    check_convert(trafo, [1, 2, 3, 4], [[1, 2], [3, 4]])

# rescale
def test_rescale_discrete():
    for s in [Discrete(10), MultiDiscrete([(0, 2), (0, 3)]), MultiBinary(5)]:
        try:
            rescale(s, 1.0)
            assert False 
        except TypeError: pass

def test_rescale_box():
    s = Box(np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    trafo = rescale(s, np.array([1.0, 0.0]), np.array([2.0, 1.0]))

    assert trafo.target == Box(np.array([1.0, 0.0]), np.array([2.0, 1.0]))
    check_convert(trafo, [1.0, 0.0], [0.0, 1.0])
    check_convert(trafo, [2.0, 1.0], [1.0, 2.0])

    # scalar rescale
    s = Box(np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    trafo = rescale(s, 0.0, 1.0)

    assert trafo.target == Box(np.array([0.0, 0.0]), np.array([1.0, 1.0]))

def test_rescale_inf():
    # check that invalid target range causes error
    try:
        rescale(Box(np.array([0.0]), np.array([1.0])), np.inf, np.inf)
    except ValueError: pass

    # positive infinite
    s = Box(np.array([0.0, 0.0]), np.array([1.0, np.inf]))
    trafo = rescale(s, np.array([1.0, 1.0]), np.array([3.0, np.inf]))

    assert trafo.target == Box(np.array([1.0, 1.0]), np.array([3.0, np.inf]))
    check_convert(trafo, [3.0, 1.0], [1.0, 0.0])

    # negative infinite
    s = Box(np.array([-1.0, -np.inf]), np.array([0.0, 0.0]))
    trafo = rescale(s, np.array([1.0, -np.inf]), np.array([3.0, 1.0]))

    assert trafo.target == Box(np.array([1.0, -np.inf]), np.array([3.0, 1.0]))
    check_convert(trafo, [1.0, 1.0], [-1.0, 0.0])

    # two sided
    s = Box(np.array([-1.0, -np.inf]), np.array([0.0, np.inf]))
    trafo = rescale(s, np.array([1.0, -np.inf]), np.array([3.0, np.inf]))

    assert trafo.target == Box(np.array([1.0, -np.inf]), np.array([3.0, np.inf]))
    check_convert(trafo, [1.0, 12.0], [-1.0, 12.0])

    # cannot linearly transform infinite to finite range
    try:
        s = Box(np.array([-np.inf, 0.0]), np.array([np.inf, np.inf]))
        trafo = rescale(s, np.array([1.0, 1.0]), np.array([3.0, np.inf]))
    except ValueError: pass




if __name__ == '__main__':
    test_discretize_1d_box()
    test_discretize_discrete()
    test_discretize_nd_box()

    test_flatten_single()
    test_flatten_discrete()
    test_flatten_continuous()

    test_rescale_discrete()
    test_rescale_box()
    test_rescale_inf()

