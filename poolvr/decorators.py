import numpy as np


def allocs_out_shape(shape, dtype=np.float64):
    def _decorator(func):
        def wrapper(*args, out=None, **kwargs):
            if out is None:
                out = np.empty(shape, dtype=np.float64)
            return func(*args, out=out, **kwargs)
        return wrapper
    return _decorator


def allocs_out(func):
    def wrapper(*args, out=None, **kwargs):
        if out is None:
            out = np.empty(3, dtype=np.float64)
        return func(*args, out=out, **kwargs)
    return wrapper


def allocs_out_vec4(func):
    def wrapper(*args, out=None, **kwargs):
        if out is None:
            out = np.empty(4, dtype=np.float64)
        return func(*args, out=out, **kwargs)
    return wrapper
