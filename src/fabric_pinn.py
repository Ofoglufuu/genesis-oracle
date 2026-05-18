"""
fabric_pinn.py — HeatSurrogate: a Flax Linen MLP for the PINN heat equation
=============================================================================

Observer-Prime, we are transcending classical grid-based solvers to build a
PINN.  This module defines a pure JAX/Flax neural network architecture that
maps continuous 2D space-time coordinates (x, t) to a 1D scalar temperature
prediction u(x, t).

Architecture
------------
    Input  : (x, t)  ∈  ℝ²
    Hidden : 4 layers × 32 neurons, tanh activation
    Output : u       ∈  ℝ¹

Why tanh?
---------
The physics loss in a PINN for the heat equation requires computing ∂²u/∂x²
via automatic differentiation.  tanh is infinitely differentiable (C∞) and its
higher-order derivatives are non-trivial (unlike ReLU, whose second derivative
is zero almost everywhere).  This makes tanh the canonical activation choice
for PINNs that enforce differential-equation constraints.

Stateless design
----------------
Following Flax conventions, the Module describes the computation graph only.
Parameters are created externally with `model.init()` and supplied explicitly
to every `model.apply()` call — keeping the network a pure function that
composes cleanly with jax.grad, jax.jit, and jax.vmap.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

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
        # the PDE residual loss requires (∂u/∂t − ∂²u/∂x² = 0).
        for i in range(self.n_layers):
            x = nn.Dense(
                features=self.hidden_dim,
                name=f"hidden_{i}",
            )(x)
            x = nn.tanh(x)

        # --- Output layer — linear projection to scalar temperature ---
        x = nn.Dense(features=self.output_dim, name="output")(x)

        return x


# ---------------------------------------------------------------------------
# Entry point — explicit initialisation & sanity check
# ---------------------------------------------------------------------------

def main():
    print("=" * 62)
    print("  HeatSurrogate — Flax Linen PINN Architecture")
    print("=" * 62)

    # ------------------------------------------------------------------
    # 1. Create a reproducible PRNG key
    # ------------------------------------------------------------------
    key = jax.random.PRNGKey(0)

    # ------------------------------------------------------------------
    # 2. Instantiate the model (architecture only — no weights yet)
    # ------------------------------------------------------------------
    model = HeatSurrogate(hidden_dim=32, n_layers=4, output_dim=1)

    # ------------------------------------------------------------------
    # 3. Build a dummy input: single (x, t) sample
    # ------------------------------------------------------------------
    dummy_input = jnp.ones((1, 2))  # shape (batch=1, features=2)

    # ------------------------------------------------------------------
    # 4. Initialise parameters with model.init()
    # ------------------------------------------------------------------
    # This traces the forward pass once, infers every layer's shape, and
    # draws random initial weights using the supplied PRNG key.
    # The returned `params` pytree lives OUTSIDE the model object.
    params = model.init(key, dummy_input)

    # ------------------------------------------------------------------
    # 5. Forward pass with model.apply()
    # ------------------------------------------------------------------
    output = model.apply(params, dummy_input)

    # ------------------------------------------------------------------
    # 6. Report
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 7. Batch forward-pass sanity check
    # ------------------------------------------------------------------
    key_batch = jax.random.PRNGKey(1)
    batch = jax.random.normal(key_batch, shape=(64, 2))
    batch_output = model.apply(params, batch)
    print(f"\n  Batch input  : {batch.shape}")
    print(f"  Batch output : {batch_output.shape}")

    print("\n" + "=" * 62)
    print("  ✓ HeatSurrogate initialised and executed successfully.")
    print("=" * 62)


if __name__ == "__main__":
    main()
