import numpy as np
from itertools import product


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    iter = product(*[list(range(0, s)) for s in x.shape])
    
    for idx in iter:
        tmp_val = x[idx]
        
        x[idx] = tmp_val + h
        fxhp = f(x)
        
        x[idx] = tmp_val - h
        fxhm = f(x)

        grad[idx] = (fxhp - fxhm) / (2 * h)
        x[idx] = tmp_val
    
    return grad


def gradient_descent(f, init_x, *, lr=0.1, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x