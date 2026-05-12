# Ascension Report

## Execution Times

| Simulation | Execution Time (s) |
|---|---|
| Legacy NumPy (`src/legacy_swarm.py`) | 0.1378 |
| JAX 2nd run (`src/jax_swarm.py`) | 0.0305 |
| **Speedup Factor** | **4.52×** |

Both simulations were run with 100,000 oscillators and 1,000 time steps
(`dt = 0.01`, `damping = 0.05`).

## Speedup Calculation

```
Speedup Factor = Legacy Time / JAX 2nd Run Time
```

```
Speedup Factor = 0.1378 / 0.0305 = 4.52×
```

## Why the First JAX Run Is Slower

The first call to a JIT-compiled JAX function is slower because JAX must
trace the Python function and compile it into an optimised XLA executable.
The second run is faster because it reuses the cached compiled executable,
bypassing the compilation overhead entirely.
