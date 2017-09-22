from gym import Wrapper, ActionWrapper
from gym import spaces
from collections import deque
import numpy as np


# In this file there are useful wrappers that are not, strictly speaking, (only) space wrappers, but
# do perform some additional work.


def RepeatActionWrapper(env, repeat):
    """
    This is just a thin wrapper around `gym.wrappes.SkipWrapper`
    to get a consistent interface.
    :param gym.env env: Environment to wrap
    :param int repeat: Number of times that an action will be repeated.
    :return gym.Wrapper: A wrapper that repeats an action for `repeat`
            steps.
    """
    from gym.wrappers import SkipWrapper
    return SkipWrapper(repeat)(env)


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
        self.observation_space = spaces.Box(low, high)

    def _step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._observations.append(obs)

        return np.stack(self._observations, axis=self._axis), rew, done, info

    def _reset(self):
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
        raise NotImplementedError()

    def _action(self, action):
        if isinstance(action, np.ndarray):
            if action.size == 1:
                return action[0]
        return action
