"""
Exercise 1 — Shattering the Grid & Anchoring Reality
=====================================================

Generate mesh-free training data for a Physics-Informed Neural Network (PINN)
solving the 1D heat equation:

    ∂u/∂t = ∂²u/∂x²

Domain:
    x ∈ [-1, 1],  t ∈ [0, 1]

Initial condition (IC):
    u(x, 0) = -sin(π·x)

Boundary conditions (BC):
    u(-1, t) = 0
    u( 1, t) = 0

Three datasets are produced:
    1. Collocation points  — random interior points where the PDE residual is enforced.
    2. IC points           — points at t=0 with known temperature from the IC.
    3. BC points           — points at x=±1 with known temperature (zero).
"""

import os
import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------

def generate_collocation_points(key: jax.Array, n_points: int = 5000) -> jax.Array:
    """Sample random (x, t) collocation points inside the space-time domain.

    These are the points where the PDE residual will be minimised during
    training.  No target temperature is required — the loss comes from the
    physics residual alone.

    Args:
        key: JAX PRNG key.
        n_points: Number of collocation points to generate.

    Returns:
        Array of shape (n_points, 2) with columns [x, t].
        x ∈ [-1, 1], t ∈ [0, 1].
    """
    key_x, key_t = jax.random.split(key)

    # Uniform samples in [-1, 1] for x
    x = jax.random.uniform(key_x, shape=(n_points, 1), minval=-1.0, maxval=1.0)
    # Uniform samples in [0, 1] for t
    t = jax.random.uniform(key_t, shape=(n_points, 1), minval=0.0, maxval=1.0)

    return jnp.concatenate([x, t], axis=1)  # (n_points, 2)


def generate_initial_condition_points(key: jax.Array, n_points: int = 500):
    """Sample IC points at t = 0 with the analytic initial temperature.

    IC: u(x, 0) = -sin(π·x)

    Args:
        key: JAX PRNG key.
        n_points: Number of IC points.

    Returns:
        inputs:  shape (n_points, 2) — columns [x, 0].
        targets: shape (n_points, 1) — u_true(x, 0).
    """
    # Random x values in [-1, 1]; t is fixed at 0
    x = jax.random.uniform(key, shape=(n_points, 1), minval=-1.0, maxval=1.0)
    t = jnp.zeros((n_points, 1))

    inputs = jnp.concatenate([x, t], axis=1)           # (n_points, 2)
    targets = -jnp.sin(jnp.pi * x)                      # (n_points, 1)

    return inputs, targets


def generate_boundary_condition_points(key: jax.Array, n_points: int = 500):
    """Sample BC points at x = -1 and x = 1 with zero Dirichlet conditions.

    BC: u(-1, t) = 0,  u(1, t) = 0

    Points are split evenly between the two boundaries (250 each for the
    default of 500 total points).

    Args:
        key: JAX PRNG key.
        n_points: Total number of BC points (split equally between x=-1 and x=1).

    Returns:
        inputs:  shape (n_points, 2) — columns [x, t].
        targets: shape (n_points, 1) — all zeros.
    """
    n_half = n_points // 2
    key_left, key_right = jax.random.split(key)

    # --- Left boundary: x = -1 ---
    t_left = jax.random.uniform(key_left, shape=(n_half, 1), minval=0.0, maxval=1.0)
    x_left = jnp.full((n_half, 1), -1.0)
    inputs_left = jnp.concatenate([x_left, t_left], axis=1)

    # --- Right boundary: x = 1 ---
    t_right = jax.random.uniform(key_right, shape=(n_points - n_half, 1),
                                  minval=0.0, maxval=1.0)
    x_right = jnp.full((n_points - n_half, 1), 1.0)
    inputs_right = jnp.concatenate([x_right, t_right], axis=1)

    # Combine both boundaries
    inputs = jnp.concatenate([inputs_left, inputs_right], axis=0)   # (n_points, 2)
    targets = jnp.zeros((n_points, 1))                              # (n_points, 1)

    return inputs, targets


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------

def generate_pinn_data(seed: int = 42):
    """Generate the complete PINN training dataset for the 1D heat equation.

    Uses explicit JAX PRNG key management so that results are fully
    reproducible for the same seed.

    Args:
        seed: Integer seed for the JAX PRNG.

    Returns:
        Dictionary with keys:
            collocation    — (5000, 2)
            ic_inputs      — (500, 2)
            ic_targets     — (500, 1)
            bc_inputs      — (500, 2)
            bc_targets     — (500, 1)
    """
    master_key = jax.random.PRNGKey(seed)

    # Split into three independent sub-keys
    key_col, key_ic, key_bc = jax.random.split(master_key, num=3)

    collocation = generate_collocation_points(key_col, n_points=5000)
    ic_inputs, ic_targets = generate_initial_condition_points(key_ic, n_points=500)
    bc_inputs, bc_targets = generate_boundary_condition_points(key_bc, n_points=500)

    return {
        "collocation": collocation,
        "ic_inputs": ic_inputs,
        "ic_targets": ic_targets,
        "bc_inputs": bc_inputs,
        "bc_targets": bc_targets,
    }


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def plot_sampling_points(data: dict, save_path: str) -> None:
    """Create a scatter plot of the three point types in the (x, t) domain.

    Args:
        data: Dictionary returned by `generate_pinn_data`.
        save_path: File path for the saved PNG figure.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    col = np.asarray(data["collocation"])
    ic  = np.asarray(data["ic_inputs"])
    bc  = np.asarray(data["bc_inputs"])

    ax.scatter(col[:, 0], col[:, 1], s=1,  alpha=0.3, label="Collocation (PDE)")
    ax.scatter(ic[:, 0],  ic[:, 1],  s=12, alpha=0.8, label="IC  (t = 0)", marker="^")
    ax.scatter(bc[:, 0],  bc[:, 1],  s=12, alpha=0.8, label="BC  (x = ±1)", marker="s")

    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title("PINN Sampling Points — 1D Heat Equation")
    ax.legend(loc="upper right")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[✓] Scatter plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Generate all datasets ---
    data = generate_pinn_data(seed=42)

    # --- Print shapes for verification ---
    print("=" * 55)
    print("PINN Training Data — 1D Heat Equation")
    print("=" * 55)
    for name, arr in data.items():
        print(f"  {name:20s}  →  shape {arr.shape}")
    print("=" * 55)

    # --- Save to .npz ---
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    npz_path = os.path.join(data_dir, "pinn_heat_data.npz")
    np.savez(
        npz_path,
        collocation=np.asarray(data["collocation"]),
        ic_inputs=np.asarray(data["ic_inputs"]),
        ic_targets=np.asarray(data["ic_targets"]),
        bc_inputs=np.asarray(data["bc_inputs"]),
        bc_targets=np.asarray(data["bc_targets"]),
    )
    print(f"[✓] Data saved → {npz_path}")

    # --- Scatter plot ---
    plot_path = os.path.join(data_dir, "pinn_sampling_points.png")
    plot_sampling_points(data, save_path=plot_path)
