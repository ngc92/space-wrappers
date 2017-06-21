import gym
from space_wrappers.transform import discretize, flatten, rescale
from gym.spaces import *
import numpy as np
import itertools

# discretize

def test_discretize_1d_box():
    cont = Box(np.array([0.0]), np.array([1.0]))
    disc, f = discretize(cont, 10)

    assert disc == Discrete(10)
    assert f(0) == 0.0
    assert f(9) == 1.0

def test_discretize_discrete():
    start = Discrete(5)
    d, f = discretize(start, 10)
    assert d == start

def test_discretize_nd_box():
    cont = Box(np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    disc, f = discretize(cont, 10)

    assert disc == MultiDiscrete([(0, 9), (0, 9)])
    assert (f((0, 0)) == [0.0, 1.0]).all()
    assert (f((9, 9)) == [1.0, 2.0]).all()

    disc, f = discretize(cont, (5, 10))

    assert disc == MultiDiscrete([(0, 4), (0, 9)])
    assert (f((0, 0)) == [0.0, 1.0]).all()
    assert (f((4, 9)) == [1.0, 2.0]).all()

# flatten
def test_flatten_single():
    start = Discrete(5)
    d, f = flatten(start)
    assert d == start

    start = Box(np.array([0.0]), np.array([1.0]))
    d, f = flatten(start)
    assert d == start

def test_flatten_discrete():
    md = MultiDiscrete([(0, 2), (0, 3)])
    d, f = flatten(md)

    assert d == Discrete(12)
    # check that we get all actions exactly once
    actions = []
    for (i, j) in itertools.product([0, 1, 2], [0, 1, 2, 3]):
        actions += [(i, j)]
    for i in range(0, 12):
        a = f(i)
        assert a in actions, (a, actions)
        actions = list(filter(lambda x: x != a, list(actions)))
    assert len(actions) == 0

    # same test for binary
    md = MultiBinary(3)
    d, f = flatten(md)

    assert d == Discrete(2**3)
    # check that we get all actions exactly once
    actions = []
    for (i, j, k) in itertools.product([0, 1], [0, 1], [0, 1]):
        actions += [(i, j, k)]
    for i in range(0, 8):
        a = f(i)
        assert a in actions, (a, actions)
        actions = list(filter(lambda x: x != a, actions))
    assert len(actions) == 0

def test_flatten_continuous():
    ct = Box(np.zeros((2,2)), np.ones((2, 2)))
    d, f = flatten(ct)

    assert d == Box(np.zeros(4), np.ones(4))
    assert (f([1, 2, 3, 4]) == [[1, 2], [3, 4]]).all()

# rescale
def test_rescale_discrete():
    for s in [Discrete(10), MultiDiscrete([(0, 2), (0, 3)]), MultiBinary(5)]:
        try:
            rescale(s, 1.0)
            assert False 
        except TypeError: pass

def test_rescale_box():
    s = Box(np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    d, f = rescale(s, np.array([1.0, 0.0]), np.array([2.0, 1.0]))

    assert d == Box(np.array([1.0, 0.0]), np.array([2.0, 1.0]))
    assert (f([1.0, 0.0]) == [0.0, 1.0]).all()
    assert (f([2.0, 1.0]) == [1.0, 2.0]).all()



if __name__ == '__main__':
    test_discretize_1d_box()
    test_discretize_discrete()
    test_discretize_nd_box()

    test_flatten_single()
    test_flatten_discrete()
    test_flatten_continuous()

    test_rescale_discrete()
    test_rescale_box()

