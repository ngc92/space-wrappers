import gym
from space_wrappers import *
from gym import spaces
import numpy as np
import pytest


class ExpectEnv(gym.Env):
    def __init__(self):
        super(ExpectEnv, self).__init__()

    def step(self, action):
        assert action == self.expectation, "{} != {}".format(action, self.expectation)


entry_point = "space_wrappers.tests.test_action_wrappers:ExpectEnv"
gym.envs.register(id='ExpectTest-v0', entry_point=entry_point)


def test_discretized_wrapper():
    expect = gym.make("ExpectTest-v0")
    cont = spaces.Box(np.array([0.0]), np.array([1.0]), dtype=np.float32)
    expect.action_space = cont
    expect.expectation = 0.5
    wrapper = DiscretizedActionWrapper(expect, 3)
    assert is_discrete(wrapper.action_space)
    wrapper.step(1)


def test_flattened_wrapper():
    expect = gym.make("ExpectTest-v0")
    md = spaces.MultiDiscrete([2, 2])
    expect.action_space = md
    expect.expectation = (1, 1)
    wrapper = FlattenedActionWrapper(expect)
    wrapper.step(3)


def test_rescaled_wrapper():
    expect = gym.make("ExpectTest-v0")
    bx = spaces.Box(np.array([0.0]), np.array([1.0]), dtype=np.float32)
    expect.action_space = bx
    expect.expectation = 0.5
    wrapper = RescaledActionWrapper(expect, np.array([1.0]), np.array([2.0]))
    wrapper.step(1.5)
