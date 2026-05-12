"""
jax_swarm.py — JAX-Accelerated Kinetic Energy Harvester Array Simulation

Translates the legacy NumPy damped harmonic oscillator swarm simulation
into idiomatic JAX.  Key differences from the legacy version:

  • Pure-function physics step vectorised with jax.vmap.
  • Time-stepping loop replaced by jax.lax.scan (no Python-level loop).
  • Entire simulation JIT-compiled via jax.jit for XLA acceleration.
  • Reproducible initialisation through jax.random.PRNGKey.
"""

import time
from functools import partial

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Core physics: one oscillator, one Euler step
# ---------------------------------------------------------------------------

def oscillator_step(x, v, w, dt=0.01, damping=0.05):
    """
    Perform a single explicit-Euler time step for one damped harmonic
    oscillator.

    Parameters
    ----------
    x : float
        Current position of the oscillator.
    v : float
        Current velocity of the oscillator.
    w : float
        Natural angular frequency of the oscillator.
    dt : float
        Integration time step (seconds).
    damping : float
        Damping coefficient.

    Returns
    -------
    new_x : float
        Updated position after the time step.
    new_v : float
        Updated velocity after the time step.
    """
    acceleration = -damping * v - w ** 2 * x
    new_x = x + dt * v
    new_v = v + dt * acceleration
    return new_x, new_v


# ---------------------------------------------------------------------------
# Vectorise the single-oscillator step over the full swarm
# ---------------------------------------------------------------------------

# vmap maps oscillator_step across the leading axis of (x, v, w), turning
# the scalar function into one that operates on arrays of oscillators.
vectorised_step = jax.vmap(oscillator_step, in_axes=(0, 0, 0, None, None))


# ---------------------------------------------------------------------------
# Full simulation (scan + jit)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("num_steps",))
def simulate_swarm(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    omega: jnp.ndarray,
    num_steps: int = 1_000,
    dt: float = 0.01,
    damping: float = 0.05,
):
    """
    Simulate the full swarm of damped harmonic oscillators.

    Uses jax.lax.scan to unroll the time-stepping loop inside XLA,
    avoiding repeated Python-level dispatch overhead.

    Parameters
    ----------
    positions : jnp.ndarray, shape (N,)
        Initial positions of all oscillators.
    velocities : jnp.ndarray, shape (N,)
        Initial velocities of all oscillators.
    omega : jnp.ndarray, shape (N,)
        Angular frequencies of all oscillators.
    num_steps : int
        Number of Euler integration steps.
    dt : float
        Time step size (seconds).
    damping : float
        Damping coefficient.

    Returns
    -------
    final_positions : jnp.ndarray, shape (N,)
    final_velocities : jnp.ndarray, shape (N,)
    """

    def scan_body(carry, _unused):
        """One time step applied to the entire swarm."""
        x, v = carry
        new_x, new_v = vectorised_step(x, v, omega, dt, damping)
        return (new_x, new_v), None          # carry forward, no per-step output

    init_carry = (positions, velocities)

    # jax.lax.scan replaces the Python for-loop with an XLA while-loop.
    # xs=None with length=num_steps drives the iteration count.
    (final_positions, final_velocities), _ = jax.lax.scan(
        scan_body, init_carry, xs=None, length=num_steps
    )

    return final_positions, final_velocities


# ---------------------------------------------------------------------------
# Random initialisation (JAX PRNG)
# ---------------------------------------------------------------------------

def initialise_state(seed: int = 42, num_oscillators: int = 100_000):
    """
    Create reproducible initial conditions for the oscillator swarm.

    Parameters
    ----------
    seed : int
        PRNG seed for reproducibility.
    num_oscillators : int
        Number of independent oscillators.

    Returns
    -------
    positions : jnp.ndarray
    velocities : jnp.ndarray
    omega : jnp.ndarray
    """
    key = jax.random.PRNGKey(seed)

    # Split the master key into three independent sub-keys
    k_omega, k_pos, k_vel = jax.random.split(key, 3)

    # Angular frequencies uniformly in [0.5, 2.0]
    omega = jax.random.uniform(k_omega, shape=(num_oscillators,),
                               minval=0.5, maxval=2.0)

    # Small random perturbations for initial state
    positions = 0.1 * jax.random.normal(k_pos, shape=(num_oscillators,))
    velocities = 0.1 * jax.random.normal(k_vel, shape=(num_oscillators,))

    return positions, velocities, omega


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    num_oscillators = 100_000
    num_steps = 1_000
    dt = 0.01
    damping = 0.05

    print("=" * 60)
    print("  Kinetic Energy Harvester Array — JAX Swarm Simulation")
    print("=" * 60)
    print(f"  Oscillators : {num_oscillators:>10,}")
    print(f"  Time steps  : {num_steps:>10,}")
    print(f"  dt          : {dt:>10.4f}")
    print(f"  Damping     : {damping:>10.4f}")
    print(f"  JAX backend : {jax.default_backend()}")
    print(f"  JAX devices : {jax.devices()}")
    print("-" * 60)

    # Reproducible initial state
    positions, velocities, omega = initialise_state(
        seed=42, num_oscillators=num_oscillators
    )

    # ------------------------------------------------------------------
    # Warm-up run — triggers JIT compilation and caches the compiled code.
    # The first call to a jit-compiled function includes compilation
    # overhead which should not be included in the timing measurement.
    # ------------------------------------------------------------------
    print("  Warm-up (JIT compilation) ...")
    warmup_pos, warmup_vel = simulate_swarm(
        positions, velocities, omega,
        num_steps=num_steps, dt=dt, damping=damping,
    )
    # Block until the warm-up computation finishes (JAX is asynchronous)
    warmup_pos.block_until_ready()
    warmup_vel.block_until_ready()

    # ------------------------------------------------------------------
    # Timed execution — uses the cached, compiled XLA executable.
    # ------------------------------------------------------------------
    print("  Running timed simulation ...")
    t_start = time.time()

    final_pos, final_vel = simulate_swarm(
        positions, velocities, omega,
        num_steps=num_steps, dt=dt, damping=damping,
    )

    # block_until_ready ensures asynchronous dispatch has completed
    # before we stop the timer.
    final_pos.block_until_ready()

    t_end = time.time()
    elapsed = t_end - t_start

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("-" * 60)
    print(f"  Mean position : {float(jnp.mean(final_pos)):>14.8f}")
    print(f"  Mean velocity : {float(jnp.mean(final_vel)):>14.8f}")
    print("-" * 60)
    print(f"  Total execution time : {elapsed:.4f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()
