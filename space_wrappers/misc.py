import gym
from gym import Wrapper, ActionWrapper
from gym import spaces
from collections import deque
import numpy as np


# In this file there are useful wrappers that are not, strictly speaking, (only) space wrappers, but
# do perform some additional work.


# This wrapper is copied from the old gym implementation.
class RepeatActionWrapper(Wrapper):
    """
        Generic common frame skipping wrapper
        Will perform action for `x` additional steps
    """

    def __init__(self, env, repeat):
        """
        RepeatActionWrapper
        :param gym.Env env: Environment to wrap
        :param int repeat: Number of times that an action will be repeated. Meaning `repeat==1` executes every action twice.
        """
        super(RepeatActionWrapper, self).__init__(env)
        self.repeat_count = repeat
        self._step_count = 0

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < (self.repeat_count + 1) and not done:
            self._step_count += 1
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            current_step += 1
        if 'skip.stepcount' in info:
            raise gym.error.Error('Key "skip.stepcount" already in info. Make sure you are not stacking '
                                  'the SkipWrapper wrappers.')
        info['skip.stepcount'] = self._step_count
        return obs, total_reward, done, info

    def _reset(self):
        self._step_count = 0
        return self.env.reset()


class StackObservationWrapper(Wrapper):
    """
    This wrapper "stacks" `count` many consecutive observations together,
    i.e. it concatenates them along a new dimension.
    For time steps when not enough observations have already happened, the remaining
    space in the observation if filled by repeating the initial state.
    Currently only works for Box spaces.
    """
    def __init__(self, env, count, axis=0):
        """
        :param gym.Env env: The environment to wrap.
        :param int count: Number of observations that should be stacked.
        :param int axis: Axis along which to stack the values.
        """
        super(StackObservationWrapper, self).__init__(env)
        self._observations = deque(maxlen=count)
        self._axis = axis
        low = env.observation_space.low
        high = env.observation_space.high
        low = np.stack([low]*count, axis=axis)
        high = np.stack([high]*count, axis=axis)
        self.observation_space = spaces.Box(low, high, dtype=env.observation_space.dtype)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._observations.append(obs)

        return np.stack(self._observations, axis=self._axis), rew, done, info

    def reset(self):
        obs = self.env.reset()
        for i in range(self._observations.maxlen):
            self._observations.append(obs)

        return np.stack(self._observations, axis=self._axis)


class ToScalarActionWrapper(ActionWrapper):
    """
    This wrapper does not change the `action_space` per se,
    but the way the action is presented to the environment.
    If the action is a scalar value in a numpy array (an
    array of size one), this scalar is extracted and given
    to the underlying env. Otherwise, the input is left
    unmodified.
    This is useful because e.g. the `Discrete` space does not
    consider numpy.array([1], dtype=int) to be valid inputs.
    Leaving the input unchanged in all other cases, instead of
    raising an error for non-scalar actions, allows this
    wrapper to be applied without having to check the underlying
    env first.
    """
    # TODO should this be extended to work with general iterables?
    def __init__(self, env):
        super(ToScalarActionWrapper, self).__init__(env)

    def _reverse_action(self, action):
        raise NotImplementedError()  # pragma: no cover

    def _action(self, action):
        if isinstance(action, np.ndarray):
            if action.size == 1:
                return action[0]
        return action


class ContinuingEnvWrapper(Wrapper):
    """
    Converts the reward signal of terminal episodes to
    a signal that corresponds to continuing interacting
    (reward rate) instead of a terminating episode.
    To that end, after a specified amount of time,
    the episode is considered terminated and a final
    reward is sent, which corresponds to the discounted
    reward that would be produced if the episode were
    to continue with a constant reward rate.
    """
    def __init__(self, env, gamma, duration):
        super(ContinuingEnvWrapper, self).__init__(env)
        self._gamma = gamma
        self._duration = duration
        self._count = 0
        self._reward = 0

    def _reset(self):
        self._count = 0
        self._reward = 0
        return self.env.reset()

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._count += 1
        self._reward += reward
        if self._count == self._duration:
            reward_rate = self._reward / self._duration
            reward = reward_rate / (1 - self._gamma)
            print("RR", self._reward, reward)
            done = True

        return obs, reward, done, info
