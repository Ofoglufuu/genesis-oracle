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

## Time Travel via Gradients

### Gradient Descent Result

Starting from `v = 10.0 m/s` with a learning rate of `0.1` and 20
iterations, the update rule `v = v - 0.1 * gradient` was applied to
minimise `(v * 5.0 - 150.0)²`.

| Parameter | Value |
|---|---|
| Target distance | 150.0 m |
| Flight time | 5.0 s |
| Optimal velocity (analytical) | 30.0 m/s |
| Velocity after 20 iterations | diverged, final printed value: −21990232555520.0 m/s |

> **Note — why divergence occurs:**
>
> For the loss function `loss = (5v − 150)²`, the gradient is:
>
> ```
> gradient = 2 · (5v − 150) · 5 = 50v − 1500
> ```
>
> Applying the required update rule:
>
> ```
> v_new = v − 0.1 · (50v − 1500)
> v_new = v − 5v + 150
> v_new = 150 − 4v
> ```
>
> The error from the optimum (`v = 30`) changes sign and grows by a
> factor of 4 each iteration, causing the velocity to diverge.
>
> The stability condition for convergence is:
>
> ```
> |1 − learning_rate · 50| < 1
> ```
>
> which requires `learning_rate < 0.04`.  The exercise-specified rate of
> `0.1` exceeds this bound, so the update rule was followed literally and
> divergence is the expected behaviour.  A smaller learning rate such as
> `0.005` would converge toward the analytical optimum of `v = 30.0 m/s`.

### What `jax.grad` Does

`jax.grad` returns a new function that computes the **exact** derivative
of the original function with respect to its argument.  It works by
tracing the computational graph built during function execution and
applying the **chain rule** automatically through every operation —
a technique known as *automatic differentiation*.

### How `jax.grad` Differs from Finite Differences

- **Finite differences** estimate the slope by evaluating the function
  at least twice with a small perturbation *h*:

  ```
  df/dv ≈ (f(v + h) - f(v)) / h
  ```

  This introduces truncation error (from a finite *h*) and is
  computationally expensive when differentiating with respect to many
  parameters.

- **`jax.grad`** uses automatic differentiation on the computational
  graph, applying the chain rule exactly through every operation in the
  function.  It does not need manual perturbations, produces
  machine-precision gradients, and scales efficiently regardless of the
  number of input parameters.

## Agentic Refactoring for the Horizon

### Explicit State in Flax vs. Implicit State in Keras

In **Keras**, model parameters are commonly stored *inside* the model
object.  After calling `model.fit()` or even just `model(x)`, the
trained weights live in `model.weights` — tightly coupled to the object
that defines the architecture.

In **Flax (Linen)**, the `nn.Module` subclass defines only the
*architecture* — which layers exist, how data flows through them, and
what activations are used.  It stores **no weights** as instance
attributes:

1. **Initialisation** — `params = model.init(key, x)` traces the
   network once to discover layer shapes and draws random initial
   weights.  The resulting `params` pytree lives *outside* the model.
2. **Forward pass** — `output = model.apply(params, x)` is a pure
   function: it takes parameters and input, returns output, and has no
   side effects.

This separation means:

- The same `model.apply` can be wrapped in `jax.jit` for compilation,
  `jax.grad` for differentiation, or `jax.vmap` for batching — all
  without special framework hooks.
- Flax separates *"what the network is"* from *"what parameter values
  it currently uses"*, making the entire training loop functionally
  pure.

### Execution Summary

`src/flax_core.py` was run successfully with `uv run python src/flax_core.py`.

| Detail | Value |
|---|---|
| Input shape | (1, 4) |
| Output shape | (1, 1) |
| Hidden layer | Dense_0 — kernel (4, 32), bias (32,) |
| Output layer | Dense_1 — kernel (32, 1), bias (1,) |
| Model output | `[[0.4506556]]` |

The output was produced by calling `model.apply(params, x)`, not by
hidden internal state — confirming that Flax treats the network as a
stateless, pure function.
