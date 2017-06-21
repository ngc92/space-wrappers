from space_wrappers.classify import *
from gym.spaces import *
import numpy as np

# is_discrete
def test_is_discrete():
    assert is_discrete(Discrete(10))
    assert is_discrete(MultiDiscrete([(0, 4), (0, 5)]))
    assert is_discrete(MultiBinary(5))
    assert is_discrete(Tuple((Discrete(5), Discrete(4))))
    assert not is_discrete(Box(np.zeros(2), np.ones(2)))

    try:
        is_discrete(5)
        assert False, "No exception throw!"
    except TypeError: pass

def test_is_compound():
    assert not is_compound(Discrete(10))
    assert is_compound(MultiDiscrete([(0, 4), (0, 5)]))
    assert is_compound(MultiBinary(5))
    assert is_compound(Tuple((Discrete(5), Discrete(4))))
    assert is_compound(Box(np.zeros(2), np.ones(2)))
    assert not is_compound(Box(np.zeros(1), np.ones(1)))

    try:
        is_compound(5)
        assert False, "No exception throw!"
    except TypeError: pass

if __name__ == '__main__':
    test_is_discrete()
    test_is_compound()