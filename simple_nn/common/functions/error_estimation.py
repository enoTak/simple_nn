import numpy as np


def mean_squared_error(y, t, one_hot=True):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    if one_hot:
        return 0.5 * np.sum((y - t)**2)
    else:
        return 0.5 * np.sum((y[:, t] - 1)**2)


d = 1e-10
def cross_entropy_error(y, t, one_hot=True):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    if one_hot:
        return -np.sum(t * np.log(y + d)) / batch_size
    else:
        return -np.sum(np.log(y[:, t])) / batch_size