import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar

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

### Run Benchmark ###
r_space = np.linspace(0.0, 4.0, 100)
lyapunov_exp = []
for r in r_space:
    times = np.linspace(0.0, 10.0, N_TIMES)
    data = gen_data_points(0.6, r, times)
    lambda_validate = validate(data, r)
    lyapunov_exp.append(lambda_validate)

plt.plot(r_space, lyapunov_exp)
plt.show()

### Perform Wolf's Algorithm ###
le_4 = lyapunov_exp[-1]
print(f"LE @ 4.0 = {le_4}")

times = np.linspace(0.0, 10.0, N_TIMES)
data = gen_data_points(0.6, r, times)

N = len(data)
d_E = 25
epsilon = 1.0E-6
embedding = np.zeros([N - d_E + 1, d_E])
for i in range(N - d_E + 1):
    embedding[i] = [data[j] for j in range(i, i+d_E)]

def edge_weight(d, dot):
    return d - dot

def nearest_neighbor(embedding, i, xz=None):
    valid_angle = np.ones(len(embedding))
    if xz is not None:
        for j, z in enumerate(embedding):
            jvec = embedding[j] - embedding[i]
            angle = np.arccos(np.dot(jvec, xz) / (np.linalg.norm(jvec) * np.linalg.norm(xz)))
            if angle < np.pi / 9.0:
                valid_angle[j] = True
            
    min_d = float('inf')
    min_j = None
    for j, z in enumerate(embedding):
        if i == j or not valid_angle[j]:
            continue
        d = np.linalg.norm(z - embedding[i])
        if d < min_d:
            min_d = d
            min_j = j
    return min_j, min_d

L = []
Lprime = []

dist = []
for i in range(len(embedding)):
    for j in range(i+1,len(embedding)):
        dist.append(np.linalg.norm(embedding[i] - embedding[j]))
epsilon = np.percentile(dist, 10)
epsilon = 2.5 * np.min(dist)

i = 0
j = 0

x = embedding[i]
j, d = nearest_neighbor(embedding, i)
L.append(d)

while d < epsilon:
    i += 1
    j += 1
    d =  np.linalg.norm(embedding[i] - embedding[j])
Lprime.append(d)

bar = Bar('Processing', max=len(embedding))

while max(i, j) < len(embedding):
    x = embedding[i]
    j, d = nearest_neighbor(embedding, i, embedding[j]-x)
    L.append(d)

    # print("hello")
    while d < epsilon and max(i, j) + 1 < len(embedding):
        # print("world")
        i += 1
        j += 1
        d =  np.linalg.norm(embedding[i] - embedding[j])
    Lprime.append(d)

    bar.next()

bar.finish()

lambda_calc = 0.0
for i in range(len(L)):
    lambda_calc += np.log2(Lprime[i] / L[i])
lambda_calc /= times[-1] - times[0]

print(f"LE_calc @ 4.0 = {lambda_calc}")