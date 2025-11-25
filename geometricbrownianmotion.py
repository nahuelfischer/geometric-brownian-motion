import numpy as np
import matplotlib.pyplot as plt

#drift coefficient
mu = 0.1
#volatility coefficient
sigma = 0.2
#initial value
S0 = 100
#time horizon
T = 10.0
#number of time steps
n = 1000
#number of paths
M = 100

dt = T / n
St = np.exp(
    (mu - sigma**2 / 2) * dt 
    + sigma * np.random.normal(0, np.sqrt(dt), size=(M, n)).T
)

St = np.vstack([np.ones(M), St])
St = S0 * St.cumprod(axis=0)

time = np.linspace(0, T, n + 1)
tt = np.full(shape=(M, n + 1), fill_value=time).T

plt.plot(tt, St)
plt.xlabel("Time")
plt.ylabel("Asset Price")
plt.title("Geometric Brownian Motion Paths")
plt.show()