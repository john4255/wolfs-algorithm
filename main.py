import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar

import random

from data import gen_data_points
from data import validate
from data import add_noise

N_TIMES = 5000

### Perform Wolf's Algorithm ###

times = np.arange(N_TIMES)
r = 3.63
data = add_noise(gen_data_points(0.6, r, times))

N = len(data)
d_E = 10
epsilon = 0.2
min_scale = 0.0001

embedding = np.zeros([N - d_E + 1, d_E])
for i in range(N - d_E + 1):
    embedding[i] = [data[j] for j in range(i, i+d_E)]

def edge_weight(d, dot):
    return d - 0.05 *dot

def nearest_neighbor(embedding, i, xz=None):
    valid_angle = np.ones(len(embedding))
    if xz is not None:
        for j, z in enumerate(embedding):
            if i == j:
                continue
            jvec = z - embedding[i]
            angle = np.arccos(np.dot(jvec, xz) / (np.linalg.norm(jvec) * np.linalg.norm(xz)))
            valid_angle[j] = angle < 0.25 * np.pi
            
    min_d = float('inf')
    min_dot = 0.0
    min_j = None

    for j, z in enumerate(embedding):
        if i == j or not valid_angle[j] or np.linalg.norm(z - embedding[i]) < min_scale:
            continue
        d = np.linalg.norm(z - embedding[i])
        dot = 0.0
        if xz is not None:
            jvec = z - embedding[i]
            dot = np.abs(np.dot(jvec, xz)) / (np.linalg.norm(jvec) * np.linalg.norm(xz))
        if edge_weight(d, dot) < edge_weight(min_d, min_dot):
            min_d = d
            min_dot = dot
            min_j = j
    return min_j, min_d

L = []
Lprime = []

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

while max(i, j) + 1 < len(embedding):
    bar.next()

    x = embedding[i]
    j1, d = nearest_neighbor(embedding, i, embedding[j]-x)
    if j1 is None:
        # L.append(np.linalg.norm(embedding[i] - embedding[j]))
        i += 1
        j += 1
        # Lprime.append(np.linalg.norm(embedding[i] - embedding[j]))
        continue
    j = j1
    L.append(d)

    while max(i, j) + 1 < len(embedding):
        i += 1
        j += 1
        d =  np.linalg.norm(embedding[i] - embedding[j])
        if d > epsilon:
            break

    Lprime.append(d)

bar.finish()

lambda_calc = 0.0
for i in range(len(L)):
    lambda_calc += np.log2(np.abs(Lprime[i] / L[i]))
lambda_calc /= len(times)

print(f"LE_calc @ {r} = {lambda_calc}")