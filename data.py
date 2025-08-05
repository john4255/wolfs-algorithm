import random
import numpy as np

def ddt_logistic_growth(x, r):
    return r - 2*r*x

def logistic_growth(x, r):
    return r * x * (1.0 - x)

def gen_data_points(x_0, r, times):
    x = x_0
    data = np.zeros(len(times))
    for i, t in enumerate(times):
        data[i] = x
        x = logistic_growth(x, r)
    return data

def add_noise(data, fraction=1.0E-6):
    return [x + fraction * (2 *random.random() - 1.0) for x in data]

def validate(data, r):
    my_lambda = 0.0

    for x in data:
        my_lambda += np.log2(np.abs(ddt_logistic_growth(x, r)))
    my_lambda /= len(data)

    return my_lambda