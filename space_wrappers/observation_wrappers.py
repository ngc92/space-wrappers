from gym import ObservationWrapper
from .transform import flatten, discretize, rescale
from .image import image_resize


class FlattenedObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super(FlattenedObservationWrapper, self).__init__(env)
        trafo = flatten(env.observation_space)
        self.observation_space = trafo.target
        self._observation = trafo.convert_to


class DiscretizedObservationWrapper(ObservationWrapper):
    def __init__(self, env, steps):
        super(DiscretizedObservationWrapper, self).__init__(env)
        trafo = discretize(env.observation_space, steps)
        self.observation_space = trafo.target
        self._observation = trafo.convert_to


class RescaledObservationWrapper(ObservationWrapper):
    def __init__(self, env, low, high):
        super(RescaledObservationWrapper, self).__init__(env)
        trafo = rescale(env.observation_space, low=low, high=high)
        self.observation_space = trafo.target
        self._observation = trafo.convert_to


class ImageResizeWrapper(ObservationWrapper):
    def __init__(self, env, new_size):
        super(ImageResizeWrapper, self).__init__(env)
        trafo = image_resize(env.observation_space, new_size=new_size)
        self.observation_space = trafo.target
        self._observation = trafo.convert_to

