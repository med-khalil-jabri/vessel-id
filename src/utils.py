import numpy as np


def norm(a):
        c = a.copy()
        c = c - np.min(c)
        max_val = np.max(c)
        c = c / max_val
        return c
