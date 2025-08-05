import numpy as np

from data import gen_data_points, validate
N_TIMES = 5000

r_space = np.linspace(0.0, 4.0, 100)
lyapunov_exp = np.array([])
for r in r_space:
    times = np.linspace(0.0, 5.0, N_TIMES)
    data = gen_data_points(0.6, r, times)
    lambda_validate = validate(data, r)
    lyapunov_exp = np.append(lyapunov_exp, lambda_validate)

le_359 = lyapunov_exp[89]
print(f"LE @ 3.59 = {le_359}")
le_363 = lyapunov_exp[90]
print(f"LE @ 3.63 = {le_363}")
le_379 = lyapunov_exp[94]
print(f"LE @ 4.79 = {le_379}")
le_4 = lyapunov_exp[99]
print(f"LE @ 4.00 = {le_4}")
