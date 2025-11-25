import cupy as cp        
import numpy as np       
import matplotlib.pyplot as plt

mu    = 0.1          # drift coefficient
sigma = 0.2          # volatility coefficient
S0    = 100.0        # initial value
T     = 10.0         # time horizon
n     = 1000         # number of time steps
M     = 100000       # number of paths

dt    = T / n        

def simulate_gbm_gpu(mu, sigma, S0, dt, n, M):
    dZ = cp.random.normal(loc=0.0,
                          scale=cp.sqrt(dt),
                          size=(M, n),
                          dtype=cp.float32)

    
    drift = (mu - 0.5 * sigma ** 2) * dt          
    diffusion = sigma * dZ                        
    
    dX = drift + diffusion                       

    dX = cp.concatenate([cp.zeros((M, 1), dtype=cp.float32), dX], axis=1)

    logSt = cp.cumsum(dX, axis=1)                  
    St    = S0 * cp.exp(logSt)                  

    return St.T                                 

gpu_paths = simulate_gbm_gpu(mu, sigma, S0, dt, n, M)  

paths_host = cp.asnumpy(gpu_paths)                    

time = np.linspace(0, T, n + 1)

plt.figure(figsize=(8, 5))
plt.plot(time, paths_host)             
plt.xlabel("Time")
plt.ylabel("Asset Price")
plt.title("Geometric Brownian Motion Paths (GPU, CuPy)")
plt.grid(True)
plt.show()