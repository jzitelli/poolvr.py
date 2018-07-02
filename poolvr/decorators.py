import numpy as np


def allocs_out(func):
    def wrapped(*args, out=None, **kwargs):
        if out is None:
            out = np.empty(3, dtype=np.float64)
        return func(*args, out=out, **kwargs)
    return wrapped


def allocs_out_vec4(func):
    def wrapped(*args, out=None, **kwargs):
        if out is None:
            out = np.empty(4, dtype=np.float64)
        return func(*args, out=out, **kwargs)
    return wrapped
