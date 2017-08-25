from numbers import Integral

import numpy as np
from gym import spaces
from .transform import Transform


# image specific transformations
def image_resize(space, new_size):
    """
    Transform an image space into one with different resolution.
    This requires that the input space is a spaces.Box space with
    rank 2. The lower and upper bounds of the transformed space are
    chosen such that the new data is guaranteed to lie within it,
    i.e. the lower bound is the minimum over all lower bounds of the
    original space, and the maximum is treated analogously.
    :param gym.Space space:
    :param int|tuple new_size: Size of the rescaled image.
    :return: A transform object describing the forward and backward
             transformations between the different image sizes. Not
             that rescaling throws away information, and thus there
             will be no perfect round-tripping.
    :rtype: Transform
    """
    if isinstance(new_size, Integral):
        new_size = (new_size, new_size)
    else:
        new_size = tuple(new_size)

    from scipy.misc import imresize

    if not isinstance(space, spaces.Box):
        raise TypeError("Only box spaces can be interpreted as images, got {}".format(space))

    source_shape = space.shape
    if len(source_shape) != 2:
        raise ValueError("Image resize expects two dimensional data, got shape {}".format(source_shape))

    def convert(data):
        return imresize(data, size=new_size)

    def back(data):
        return imresize(data, size=source_shape)

    return Transform(space, spaces.Box(np.min(space.low), np.max(space.high), shape=new_size), convert, back)


# TODO resample, grayscale, cropping