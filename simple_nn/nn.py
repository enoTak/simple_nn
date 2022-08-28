__all__ = [
    "init_network", "predict"]


import pickle
import numpy as np
from itertools import product


from .common import functions as func


def init_network(pkl=None):
    network = {}
    
    if pkl:
        with open(pkl, "rb") as f:
            network = pickle.load(f)

    else:
        network["W1"] = np.random.randn(2, 3)
        network["b1"] = np.random.randn(3)
        network["W2"] = np.random.randn(3, 2)
        network["b2"] = np.random.randn(2)
        network["W3"] = np.random.randn(2, 2)
        network["b3"] = np.random.randn(2)

    return network


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    
    a1 = x @ W1 + b1
    z1 = func.sigmoid(a1)
    a2 = z1 @ W2 + b2
    z2 = func.sigmoid(a2)
    a3 = z2 @ W3 + b3
    y = func.softmax(a3)
    
    return y
