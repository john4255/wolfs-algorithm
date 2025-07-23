import numpy as np
import matplotlib.pyplot as plt

import random

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

def gen_noisy_data(data, fraction=0.01):
    return [x + fraction * (2 *random.random() - 1.0) for x in data]

def validate(data, r):
    my_lambda = 0.0

    for x in data:
        my_lambda += np.log2(np.abs(ddt_logistic_growth(x, r))) / len(data)

    return my_lambda

r_space = np.linspace(0.0, 4.0, 1000)
lyapunov_exp = []
for r in r_space:
    times = np.linspace(0.0, 25.0, 1000)
    data = gen_data_points(0.6, r, times)
    my_lambda = validate(data, r)
    lyapunov_exp.append(my_lambda)

plt.plot(r_space, lyapunov_exp)
plt.show()