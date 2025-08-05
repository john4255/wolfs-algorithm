import numpy as np
import matplotlib.pyplot as plt

from data import gen_data_points
from data import validate
from data import add_noise

N_TIMES = 5000

### Run Benchmark ###
r_space = np.linspace(0.0, 4.0, 100)
lyapunov_exp = []
for r in r_space:
    times = np.linspace(0.0, 5.0, N_TIMES)
    data = gen_data_points(0.6, r, times)
    lambda_validate = validate(data, r)
    lyapunov_exp.append(lambda_validate)
plt.plot(r_space, lyapunov_exp)

plt.title("Lyapunov Exponents for Logistic Growth Map")
plt.xlabel("r")
plt.ylabel("LE")
plt.show()
