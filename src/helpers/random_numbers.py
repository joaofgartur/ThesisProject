import random
import numpy as np
from sklearn.utils import check_random_state

numpy_generator = None
random_seed = None


def set_generator(seed: int):
    return np.random.default_rng(seed)


def get_generator():
    return numpy_generator


def set_seed(seed: int):
    global random_seed
    global numpy_generator

    random_seed = seed
    numpy_generator = set_generator(seed)

    # random module
    random.seed(seed)

    # numpy
    np.random.seed(seed)

    # sklearn
    random_state = check_random_state(seed)
    random_state.seed(seed)


def get_seed():
    return random_seed
