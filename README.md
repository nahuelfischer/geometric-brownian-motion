geometric-brownian-motion


Geometric Brownian Motion Simulation (CPU & GPU)

This repository contains two Python implementations for simulating Geometric Brownian Motion (GBM), a widely used stochastic process in quantitative finance for modeling asset prices.

1. CPU Version (NumPy)

The first script uses NumPy to generate and plot multiple GBM paths.
It computes stochastic price trajectories based on a drift term, volatility, and normally distributed random shocks.
This version is ideal for smaller simulations or environments without GPU support.

Features:

- Straightforward NumPy implementation
- Generates 𝑀 paths with 𝑛 time steps
- Uses cumulative products to compute GBM evolution
- Plots the resulting asset-price paths using Matplotlib

2. GPU-Accelerated Version (CuPy)

The second script replicates the GBM simulation using CuPy, enabling massive performance gains through NVIDIA CUDA GPUs.
By generating random increments and performing vectorized operations directly on the GPU, it supports simulations with hundreds of thousands of paths while remaining computationally efficient.

Features:

- Fully GPU-accelerated GBM simulation
- Efficient random number generation with CuPy
- Handles very large path counts (e.g., 100,000+)
- Transfers results back to the host (NumPy) for plotting
- Produces the same mathematical model but at significantly higher speed

Requirements:

This version requires an NVIDIA GPU and a proper CUDA installation.
See the CuPy installation guide for details:
https://docs.cupy.dev/en/stable/install.html

Warning:

Setting n and M too high may result in very large VRAM and RAM usage, and in some cases even temporary storage usage.
It is recommended to increase n and M gradually to avoid memory issues.
