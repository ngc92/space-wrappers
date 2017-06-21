import gym
from space_wrappers import *
from gym import spaces
import numpy as np

class ProvideEnv(gym.Env):
    def __init__(self):
        super(ProvideEnv, self).__init__()

    def _step(self, action):
        return self.provide_observation, 0.0, True, {}

entry_point = "space_wrappers.tests.test_observation_wrappers:ProvideEnv"

def test_discretized_wrapper():
    expect = gym.make("ProvideTest-v0")
    cont = spaces.Box(np.array([0.0]), np.array([1.0]))
    expect.observation_space = cont
    expect.provide_observation = 0.5
    wrapper = DiscretizedObservationWrapper(expect, 3)
    assert is_discrete(wrapper.observation_space)
    o, r, d, i = wrapper.step(1)
    assert wrapper.observation_space.contains(o)
    assert o == 1

def test_flattened_wrapper():
    expect = gym.make("ProvideTest-v0")
    md = spaces.MultiDiscrete([(0, 1), (0, 1)])
    expect.observation_space = md
    expect.provide_observation  = (1, 1)
    wrapper = FlattenedObservationWrapper(expect)
    o, r, d, i = wrapper.step(3)
    assert wrapper.observation_space.contains(o)
    assert o == 3

def test_rescaled_wrapper():
    expect = gym.make("ProvideTest-v0")
    bx = spaces.Box(np.array([0.0]), np.array([1.0]))
    expect.observation_space = bx
    expect.provide_observation  = 0.5
    wrapper = RescaledObservationWrapper(expect, np.array([1.0]), np.array([2.0]))
    o, r, d, i = wrapper.step(1.5)
    assert wrapper.observation_space.contains(o)
    assert o == 1.5


if __name__ == '__main__':
    gym.envs.register(id='ProvideTest-v0', entry_point=entry_point)
    test_discretized_wrapper()
    test_flattened_wrapper()
    test_rescaled_wrapper()
