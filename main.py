import numpy as np
import matplotlib.pyplot as plt

import random

N_TIMES = 1000

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
        my_lambda += np.log2(np.abs(ddt_logistic_growth(x, r)))
    my_lambda /= len(data)

    return my_lambda

def nearest_neighbor(embedding, z1, z2=None):
    min_d = float('inf')
    min_angle = float('inf')
    min_z = embedding[0]
    for z in embedding:
        d = np.linalg.norm(z - z1)
        if z2 is not None: # Preserve angle as much as possible
            angle = np.arccos(np.dot(z - z2, z1 - z2) / (np.linalg.norm(z - z1) * np.linalg.norm(z - z2)))
        else:
            angle = 0.0
        if d < 0.001:
            continue
        if z2 is not None and np.linalg.norm(z - z2) < 0.001:
            continue
        if d + 1.9 * angle < min_d + 1.9 * min_angle:
            min_d = d
            min_angle = angle
            min_z = z
    return min_z, min_d

def evolve(x, r):
    return x + logistic_growth(x, r)

r_space = np.linspace(0.0, 4.0, 100)
lyapunov_exp = []
for r in r_space:
    times = np.linspace(0.0, 25.0, N_TIMES)
    data = gen_data_points(0.6, r, times)
    lambda_validate = validate(data, r)
    lyapunov_exp.append(lambda_validate)

plt.plot(r_space, lyapunov_exp)
plt.show()

le_4 = lyapunov_exp[-1]
print(f"LE @ 4.0 = {le_4}")

# r = 4.0
times = np.linspace(0.0, 25.0, N_TIMES)
data = gen_data_points(0.6, r, times)
# data = gen_noisy_data(data)

N = len(data)
d_E = 25
epsilon = 1.0E-6
embedding = np.zeros([N - d_E + 1, d_E])
for i in range(N - d_E + 1):
    embedding[i] = [data[j] for j in range(i, i+d_E)]

L = []
Lprime = []

x = embedding[0]
z, d = nearest_neighbor(embedding, x)
for i, x in enumerate(embedding):
    if i % 10 == 0:
        print(f"{i}/{N-d_E+1}")

    L.append(d)

    x = evolve(x, r)
    z = evolve(z, r)
    z, d = nearest_neighbor(embedding, z, x)
    Lprime.append(np.linalg.norm(x - z))

lambda_calc = 0.0
for i in range(len(L)):
    lambda_calc += np.log2(Lprime[i] / L[i])
lambda_calc /= times[-1] - times[0]

print(f"LE_calc @ 4.0 = {lambda_calc}")