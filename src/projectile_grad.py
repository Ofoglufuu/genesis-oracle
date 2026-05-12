"""
projectile_grad.py — Gradient-Based Projectile Optimisation with JAX

Demonstrates JAX's automatic differentiation (jax.grad) by optimising
the initial velocity of a 1D projectile so that it reaches a target
horizontal distance of 150 m after exactly 5 seconds of flight.

The loss function is pure and side-effect-free, enabling JAX to
differentiate through it exactly via the chain rule.
"""

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Pure loss function
# ---------------------------------------------------------------------------

def projectile_loss(v_initial):
    """
    Compute the Mean Squared Error between the simulated distance
    and the target distance.

    Parameters
    ----------
    v_initial : float
        Initial horizontal velocity (m/s).

    Returns
    -------
    loss : float
        (distance - target) ** 2
    """
    # Simple 1D model: constant velocity over 5.0 seconds
    distance = v_initial * 5.0

    # Strict target distance
    target = 150.0

    # Mean Squared Error (single sample)
    loss = (distance - target) ** 2
    return loss


# ---------------------------------------------------------------------------
# Gradient function via automatic differentiation
# ---------------------------------------------------------------------------

# jax.grad computes the exact derivative of projectile_loss with respect
# to its first (and only) argument, v_initial.
grad_loss = jax.grad(projectile_loss)


# ---------------------------------------------------------------------------
# Entry point — gradient descent loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Projectile Gradient Descent — JAX Automatic Differentiation")
    print("=" * 60)

    # Starting guess for initial velocity
    v = 10.0

    # Hyperparameters
    learning_rate = 0.1
    num_iterations = 20

    print(f"  Target distance : 150.0 m")
    print(f"  Flight time     : 5.0 s")
    print(f"  Initial v       : {v}")
    print(f"  Learning rate   : {learning_rate}")
    print(f"  Iterations      : {num_iterations}")
    print("-" * 60)
    print(f"  {'Iter':>4s}  {'Velocity':>12s}  {'Loss':>14s}  {'Gradient':>12s}")
    print("-" * 60)

    for i in range(num_iterations):
        loss = projectile_loss(v)
        gradient = grad_loss(v)

        # Log current state
        print(f"  {i:4d}  {float(v):12.6f}  {float(loss):14.6f}  {float(gradient):12.6f}")

        # Gradient descent update
        v = v - learning_rate * gradient

    # Final result
    print("-" * 60)
    print(f"  Optimised velocity : {float(v):.6f} m/s")
    print(f"  Final loss         : {float(projectile_loss(v)):.6f}")
    print(f"  Final distance     : {float(v * 5.0):.6f} m")
    print("=" * 60)


if __name__ == "__main__":
    main()
