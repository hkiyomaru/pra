"""Utility functions."""
import logging

import random
import numpy as np


def init_logger(name='logger'):
    global logger

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)

    return logger


def set_random_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
