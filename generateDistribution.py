from random import choice
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import os.path


def get_quantiles(pd, mu, sigma):

    discrim = -2 * sigma**2 * np.log(pd * sigma * np.sqrt(2 * np.pi))

    # no real roots
    if discrim < 0:
        return None

    # one root, where x == mu
    elif discrim == 0:
        return mu

    # two roots
    else:
        return choice([mu - np.sqrt(discrim), mu + np.sqrt(discrim)])