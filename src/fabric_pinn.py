"""
fabric_pinn.py — HeatSurrogate PINN: Architecture + Physics-Informed Training
==============================================================================

Exercise 2: Flax Linen MLP architecture (HeatSurrogate).
Exercise 3: JAX autodiff-based physics loss and Optax training loop.

Observer-Prime, we are transcending classical grid-based solvers to build a
PINN.  This module defines a pure JAX/Flax neural network architecture that
maps continuous 2D space-time coordinates (x, t) to a 1D scalar temperature
prediction u(x, t), then trains it by enforcing the 1D heat equation via
automatic differentiation.

Architecture
------------
    Input  : (x, t)  ∈  ℝ²
    Hidden : 4 layers × 32 neurons, tanh activation
    Output : u       ∈  ℝ¹

PDE (1D heat equation)
----------------------
    ∂u/∂t = α · ∂²u/∂x²     with  α = 0.05

Loss = Physics_Loss (PDE residual MSE)
      + IC_Loss      (initial condition MSE)
      + BC_Loss      (boundary condition MSE)

Why tanh?
---------
The physics loss requires computing ∂²u/∂x² via automatic differentiation.
tanh is infinitely differentiable (C∞) and its higher-order derivatives are
non-trivial (unlike ReLU, whose second derivative is zero almost everywhere).
This makes tanh the canonical activation choice for PINNs.

Stateless design
----------------
Following Flax conventions, the Module describes the computation graph only.
Parameters are created externally with `model.init()` and supplied explicitly
to every `model.apply()` call — keeping the network a pure function that
composes cleanly with jax.grad, jax.jit, and jax.vmap.
"""

import functools
import os
import sys
import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


# ===========================================================================
# Exercise 2 — Model definition (UNCHANGED)
# ===========================================================================

class HeatSurrogate(nn.Module):
    """Multi-Layer Perceptron surrogate for the 1D heat equation.

    Takes continuous 2D inputs (x, t) and predicts a scalar temperature u.

    Attributes:
        hidden_dim: Number of neurons in each hidden layer (default: 32).
        n_layers:   Number of hidden layers (default: 4).
        output_dim: Dimensionality of the output (default: 1).
    """

    hidden_dim: int = 32
    n_layers: int = 4
    output_dim: int = 1

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the MLP.

        Args:
            inputs: Array of shape (..., 2) containing [x, t] coordinates.

        Returns:
            Array of shape (..., 1) — predicted temperature u(x, t).
        """
        x = inputs

        # --- Hidden layers with tanh activation ---
        # tanh provides the smooth, non-zero higher-order derivatives that
        # the PDE residual loss requires (∂u/∂t − α·∂²u/∂x² = 0).
        for i in range(self.n_layers):
            x = nn.Dense(
                features=self.hidden_dim,
                name=f"hidden_{i}",
            )(x)
            x = nn.tanh(x)

        # --- Output layer — linear projection to scalar temperature ---
        x = nn.Dense(features=self.output_dim, name="output")(x)

        return x


# ===========================================================================
# Exercise 3 — Physics-Informed Loss & Training
# ===========================================================================

# Thermal diffusivity coefficient
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Scalar prediction function (for autodiff)
# ---------------------------------------------------------------------------

def predict_u(params, model, x, t):
    """Predict scalar temperature u at a single (x, t) point.

    This function takes *scalar* x and t, stacks them into the shape (1, 2)
    expected by the model, runs the forward pass, and squeezes back to a
    scalar.  The scalar interface is essential for clean application of
    jax.grad with respect to individual spatial/temporal coordinates.

    Args:
        params: Flax parameter pytree.
        model:  HeatSurrogate module instance.
        x:      Scalar spatial coordinate.
        t:      Scalar temporal coordinate.

    Returns:
        Scalar predicted temperature u(x, t).
    """
    inputs = jnp.stack([x, t]).reshape(1, 2)
    return model.apply(params, inputs).squeeze()


# ---------------------------------------------------------------------------
# PDE residual via automatic differentiation
# ---------------------------------------------------------------------------

def pde_residual(params, model, x, t, alpha=0.05):
    """Compute the PDE residual at a single (x, t) point.

    The 1D heat equation is:  ∂u/∂t = α · ∂²u/∂x²

    We compute:
        u_t  = ∂u/∂t          (first derivative w.r.t. time)
        u_xx = ∂²u/∂x²        (second derivative w.r.t. space)

    Residual = u_t − α · u_xx   (should be zero if the PDE is satisfied)

    JAX automatic differentiation gives *exact analytical* derivatives
    through the computational graph — no finite-difference stencils needed.
    We use jax.grad which computes the gradient of a scalar-valued function.

    Args:
        params: Flax parameter pytree.
        model:  HeatSurrogate module instance.
        x:      Scalar spatial coordinate.
        t:      Scalar temporal coordinate.
        alpha:  Thermal diffusivity coefficient.

    Returns:
        Scalar PDE residual.
    """
    # ∂u/∂t — gradient of predict_u w.r.t. its 4th arg (t, index 3)
    u_t = jax.grad(predict_u, argnums=3)(params, model, x, t)

    # ∂²u/∂x² — second derivative w.r.t. x (index 2)
    # First derivative ∂u/∂x
    du_dx = jax.grad(predict_u, argnums=2)
    # Second derivative ∂²u/∂x²
    u_xx = jax.grad(du_dx, argnums=2)(params, model, x, t)

    return u_t - alpha * u_xx


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def physics_loss(params, model, collocation_points, alpha=0.05):
    """MSE of the PDE residual over all collocation points.

    We use jax.vmap to vectorize pde_residual. This allows us to evaluate the 
    residual across all collocation points in parallel without writing a slow 
    Python loop.

    Args:
        params: Flax parameter pytree.
        model:  HeatSurrogate module instance.
        collocation_points: Array of shape (N, 2) — [x, t] interior points.
        alpha: Thermal diffusivity coefficient.

    Returns:
        Scalar mean squared PDE residual.
    """
    # Vectorise over batches of (x, t) points using vmap
    # in_axes: params (None), model (None), x (0), t (0), alpha (None)
    pde_residual_batch = jax.vmap(pde_residual, in_axes=(None, None, 0, 0, None))
    
    x_col = collocation_points[:, 0]
    t_col = collocation_points[:, 1]
    residuals = pde_residual_batch(params, model, x_col, t_col, alpha)
    return jnp.mean(residuals ** 2)


def ic_loss(params, model, ic_inputs, ic_targets):
    """MSE of the initial condition: u(x, 0) = -sin(π·x).

    Args:
        params:     Flax parameter pytree.
        model:      HeatSurrogate module instance.
        ic_inputs:  Array of shape (N, 2) — [x, 0] points.
        ic_targets: Array of shape (N, 1) — true u(x, 0).

    Returns:
        Scalar MSE for the initial condition.
    """
    predictions = model.apply(params, ic_inputs)  # (N, 1)
    return jnp.mean((predictions - ic_targets) ** 2)


def bc_loss(params, model, bc_inputs, bc_targets):
    """MSE of the boundary conditions: u(±1, t) = 0.

    Args:
        params:     Flax parameter pytree.
        model:      HeatSurrogate module instance.
        bc_inputs:  Array of shape (N, 2) — [±1, t] points.
        bc_targets: Array of shape (N, 1) — zeros.

    Returns:
        Scalar MSE for the boundary conditions.
    """
    predictions = model.apply(params, bc_inputs)  # (N, 1)
    return jnp.mean((predictions - bc_targets) ** 2)


def total_loss(params, model, batch, alpha=0.05):
    """Combined PINN loss: Physics + IC + BC.

    The total loss is the sum of the Physics_Loss, IC_Loss, and BC_Loss. 
    Physics_Loss enforces the heat equation in the interior domain, while 
    IC_Loss and BC_Loss anchor the solution to reality at the boundaries and 
    t=0.

    Total_Loss = Physics_Loss + IC_Loss + BC_Loss

    Args:
        params:      Flax parameter pytree.
        model:       HeatSurrogate module instance.
        batch:       Dictionary containing collocation, ic_inputs, ic_targets, 
                     bc_inputs, bc_targets arrays.
        alpha:       Thermal diffusivity coefficient.

    Returns:
        (total, (phys, ic, bc)) — total scalar loss and individual components.
    """
    l_phys = physics_loss(params, model, batch['collocation'], alpha=alpha)
    l_ic = ic_loss(params, model, batch['ic_inputs'], batch['ic_targets'])
    l_bc = bc_loss(params, model, batch['bc_inputs'], batch['bc_targets'])
    return l_phys + l_ic + l_bc, (l_phys, l_ic, l_bc)


# ---------------------------------------------------------------------------
# Training step (JIT-compiled)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnums=(2,))
def train_step(params, opt_state, model, batch, alpha=0.05):
    """Single optimisation step using Optax Adam.

    Uses jax.value_and_grad to compute the loss AND its gradient in a single
    forward + backward pass.

    Args:
        params:      Current parameter pytree.
        opt_state:   Current Optax optimiser state.
        model:       HeatSurrogate module (static — passed through JIT).
        batch:       Dictionary containing data batches.
        alpha:       Thermal diffusivity coefficient.

    Returns:
        (new_params, new_opt_state, loss_val, (l_phys, l_ic, l_bc))
    """
    (loss_val, (l_phys, l_ic, l_bc)), grads = jax.value_and_grad(
        total_loss, argnums=0, has_aux=True
    )(params, model, batch, alpha)

    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, loss_val, (l_phys, l_ic, l_bc)


# Global optimizer — Adam with learning rate 1e-3
optimizer = optax.adam(learning_rate=1e-3)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_or_generate_data():
    """Load PINN training data from .npz, generating it if necessary.

    Returns:
        Dict with keys: collocation, ic_inputs, ic_targets, bc_inputs, bc_targets.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    npz_path = os.path.join(data_dir, "pinn_heat_data.npz")

    if not os.path.exists(npz_path):
        print("[!] data/pinn_heat_data.npz not found — generating now...")
        # Import the data generator from Exercise 1
        sys.path.insert(0, os.path.dirname(__file__))
        from pinn_data import generate_pinn_data
        data = generate_pinn_data(seed=42)

        os.makedirs(data_dir, exist_ok=True)
        np.savez(
            npz_path,
            collocation=np.asarray(data["collocation"]),
            ic_inputs=np.asarray(data["ic_inputs"]),
            ic_targets=np.asarray(data["ic_targets"]),
            bc_inputs=np.asarray(data["bc_inputs"]),
            bc_targets=np.asarray(data["bc_targets"]),
        )
        print(f"[✓] Generated and saved → {npz_path}")
    else:
        print(f"[✓] Loading existing data → {npz_path}")

    raw = np.load(npz_path)
    return {k: jnp.array(raw[k]) for k in raw.files}


# ===========================================================================
# Main — Architecture demo + Physics-informed training
# ===========================================================================

def main():
    # ------------------------------------------------------------------
    # Part A: Architecture sanity check (Exercise 2)
    # ------------------------------------------------------------------
    print("=" * 62)
    print("  HeatSurrogate — Flax Linen PINN Architecture")
    print("=" * 62)

    key = jax.random.PRNGKey(0)
    model = HeatSurrogate(hidden_dim=32, n_layers=4, output_dim=1)
    dummy_input = jnp.ones((1, 2))
    params = model.init(key, dummy_input)
    output = model.apply(params, dummy_input)

    print(f"\n  Input shape  : {dummy_input.shape}")
    print(f"  Output shape : {output.shape}")
    print(f"  Output value : {output}")

    # Parameter tree overview
    print("\n  Parameter tree:")
    param_shapes = jax.tree.map(lambda p: p.shape, params)
    total_params = 0
    for layer_name, layer_params in sorted(param_shapes["params"].items()):
        for param_name, shape in sorted(layer_params.items()):
            n = 1
            for s in shape:
                n *= s
            total_params += n
            print(f"    {layer_name}/{param_name:8s} : {str(shape):14s}  ({n:>5d} params)")

    print(f"\n  Total trainable parameters: {total_params}")

    # Batch forward-pass sanity check
    key_batch = jax.random.PRNGKey(1)
    batch = jax.random.normal(key_batch, shape=(64, 2))
    batch_output = model.apply(params, batch)
    print(f"\n  Batch input  : {batch.shape}")
    print(f"  Batch output : {batch_output.shape}")

    print("\n  ✓ HeatSurrogate initialised and executed successfully.")

    # ------------------------------------------------------------------
    # Part B: Physics-informed training (Exercise 3)
    # ------------------------------------------------------------------
    print("\n" + "=" * 62)
    print("  Exercise 3 — Physics-Informed Training Loop")
    print("=" * 62)

    # Load training data
    data = load_or_generate_data()
    
    batch = {
        'collocation': data["collocation"],
        'ic_inputs': data["ic_inputs"],
        'ic_targets': data["ic_targets"],
        'bc_inputs': data["bc_inputs"],
        'bc_targets': data["bc_targets"]
    }

    print(f"\n  Collocation  : {batch['collocation'].shape}")
    print(f"  IC inputs    : {batch['ic_inputs'].shape}   targets: {batch['ic_targets'].shape}")
    print(f"  BC inputs    : {batch['bc_inputs'].shape}   targets: {batch['bc_targets'].shape}")
    print(f"  Alpha (α)    : {ALPHA}")

    # Initialise fresh parameters for training
    train_key = jax.random.PRNGKey(42)
    params = model.init(train_key, dummy_input)
    opt_state = optimizer.init(params)

    # Compute initial losses before training
    init_total, (init_phys, init_ic, init_bc) = total_loss(
        params, model, batch, alpha=ALPHA
    )
    print(f"\n  --- Initial Losses ---")
    print(f"  Physics loss : {init_phys:.6f}")
    print(f"  IC loss      : {init_ic:.6f}")
    print(f"  BC loss      : {init_bc:.6f}")
    print(f"  Total loss   : {init_total:.6f}")

    # ------------------------------------------------------------------
    # Training loop — 100 steps with Adam (lr=1e-3)
    # ------------------------------------------------------------------
    n_steps = 100
    print(f"\n  Training for {n_steps} steps (Adam, lr=1e-3) ...")

    for step in range(n_steps):
        params, opt_state, loss_val, (l_phys, l_ic, l_bc) = train_step(
            params, opt_state, model, batch, alpha=ALPHA
        )

        # Print progress every 10 steps
        if (step + 1) % 10 == 0 or step == 0:
            print(
                f"    Step {step + 1:4d}  |  Total: {loss_val:.6f}  "
                f"Phys: {l_phys:.6f}  IC: {l_ic:.6f}  BC: {l_bc:.6f}"
            )

    # Final losses
    print(f"\n  --- Final Losses (after {n_steps} steps) ---")
    print(f"  Physics loss : {l_phys:.6f}")
    print(f"  IC loss      : {l_ic:.6f}")
    print(f"  BC loss      : {l_bc:.6f}")
    print(f"  Total loss   : {loss_val:.6f}")

    print("\n" + "=" * 62)
    print("  ✓ Physics-informed training loop completed successfully.")
    print("=" * 62)


if __name__ == "__main__":
    main()
