"""
flax_core.py — Stateless MLP with Flax Linen

Demonstrates how Flax explicitly separates model *definition* from
parameter *creation* and *usage*, making the neural network a pure
function — ideal for JAX transformations (jit, grad, vmap).

Key contrast with Keras:
  • Keras: weights live *inside* the model object (model.weights).
  • Flax:  the Module describes the architecture only; parameters are
           created externally via model.init() and passed into
           model.apply() for every forward pass.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn


# ---------------------------------------------------------------------------
# Model definition — architecture only, NO stored weights
# ---------------------------------------------------------------------------

class SimpleMLP(nn.Module):
    """
    A minimal Multi-Layer Perceptron with one hidden layer.

    The @nn.compact decorator allows us to define sub-layers inline
    inside __call__ rather than in a separate setup() method.

    IMPORTANT: This class describes *what the network computes*, not
    *what values the parameters currently hold*.  The Module itself
    is stateless — it never stores trained weights as instance
    attributes the way a Keras model would.
    """

    hidden_dim: int = 32      # width of the hidden layer
    output_dim: int = 1       # width of the output layer

    @nn.compact
    def __call__(self, x):
        # Hidden layer — Dense linear transformation followed by ReLU
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)

        # Output layer — linear projection to the final dimensionality
        x = nn.Dense(features=self.output_dim)(x)
        return x


# ---------------------------------------------------------------------------
# Entry point — explicit initialisation and forward pass
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Flax Stateless MLP — Explicit State Management Demo")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Create a reproducible PRNG key
    # ------------------------------------------------------------------
    # JAX requires explicit PRNG keys for any random operation.
    # This key will be consumed by model.init() to draw the initial
    # random parameter values.
    key = jax.random.PRNGKey(42)

    # ------------------------------------------------------------------
    # 2. Build a dummy input tensor
    # ------------------------------------------------------------------
    # Shape (1, 4): one sample with four features.
    x = jnp.ones((1, 4))

    # ------------------------------------------------------------------
    # 3. Instantiate the model (architecture only)
    # ------------------------------------------------------------------
    # At this point, NO parameters exist yet.  The model object is a
    # lightweight descriptor of the computation graph — analogous to
    # a blueprint, not a trained artifact.
    model = SimpleMLP(hidden_dim=32, output_dim=1)

    # ------------------------------------------------------------------
    # 4. Initialise parameters EXTERNALLY with model.init()
    # ------------------------------------------------------------------
    # model.init(key, x) traces the forward pass once to discover every
    # layer's shape, then draws random initial weights using `key`.
    #
    # The returned `params` is a nested dict (a "pytree") that lives
    # OUTSIDE the model object.  The model never secretly caches these
    # weights — this is the fundamental difference from Keras, where
    # model.weights stores mutable parameter tensors inside the object.
    params = model.init(key, x)

    # ------------------------------------------------------------------
    # 5. Run the forward pass EXPLICITLY with model.apply()
    # ------------------------------------------------------------------
    # model.apply(params, x) is a pure function call:
    #   • It receives the parameter tree and the input.
    #   • It returns the output.
    #   • It has NO side effects and NO hidden mutable state.
    #
    # In Keras you would write `output = model(x)` and the weights
    # used are implicitly the ones stored in `model.weights`.
    # In Flax you ALWAYS pass the parameters explicitly.
    output = model.apply(params, x)

    # ------------------------------------------------------------------
    # 6. Print results
    # ------------------------------------------------------------------
    print(f"\n  Input shape  : {x.shape}")
    print(f"  Output shape : {output.shape}")

    print("\n  Parameter tree structure:")
    # jax.tree.map extracts the shape of every leaf in the pytree,
    # giving a compact overview of the network's learnable tensors.
    param_shapes = jax.tree.map(lambda p: p.shape, params)
    for layer_name, layer_params in param_shapes["params"].items():
        for param_name, shape in layer_params.items():
            print(f"    {layer_name}/{param_name} : {shape}")

    print(f"\n  Model output : {output}")
    print("\n" + "-" * 60)
    print("  The output was produced by model.apply(params, x).")
    print("  The model object itself stores NO weights — params")
    print("  are created and managed externally at all times.")
    print("=" * 60)


if __name__ == "__main__":
    main()
