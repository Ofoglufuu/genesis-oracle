"""
legacy_swarm.py — Massive Kinetic Energy Harvester Array Simulation

Simulates 100,000 independent damped harmonic oscillators using a basic
explicit Euler integration scheme over 1,000 discrete time steps.
Uses only pure Python, NumPy, and Python's built-in time module.
"""

import time
import numpy as np


def simulate_swarm(
    num_oscillators: int = 100_000,
    num_steps: int = 1_000,
    dt: float = 0.01,
    damping: float = 0.05,
    seed: int = 42,
):
    """
    Simulate an array of damped harmonic oscillators via explicit Euler.

    Parameters
    ----------
    num_oscillators : int
        Number of independent oscillators in the harvester array.
    num_steps : int
        Number of discrete time steps to integrate.
    dt : float
        Integration time step (seconds).
    damping : float
        Damping coefficient applied to each oscillator's velocity.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    positions : np.ndarray
        Final position of every oscillator after integration.
    velocities : np.ndarray
        Final velocity of every oscillator after integration.
    """
    rng = np.random.default_rng(seed)

    # Natural angular frequencies drawn from a uniform distribution
    omega = rng.uniform(0.5, 2.0, size=num_oscillators)

    # Initial conditions — small random perturbations
    positions = 0.1 * rng.standard_normal(num_oscillators)
    velocities = 0.1 * rng.standard_normal(num_oscillators)

    # Pre-compute omega squared once
    omega_sq = omega ** 2

    # Explicit (forward) Euler time-stepping loop
    for _ in range(num_steps):
        acceleration = -damping * velocities - omega_sq * positions

        new_positions = positions + dt * velocities
        new_velocities = velocities + dt * acceleration

        positions = new_positions
        velocities = new_velocities

    return positions, velocities


def main():
    num_oscillators = 100_000
    num_steps = 1_000

    print("=" * 60)
    print("  Kinetic Energy Harvester Array — Legacy Swarm Simulation")
    print("=" * 60)
    print(f"  Oscillators : {num_oscillators:>10,}")
    print(f"  Time steps  : {num_steps:>10,}")
    print(f"  dt          : {0.01:>10.4f}")
    print(f"  Damping     : {0.05:>10.4f}")
    print("-" * 60)

    t_start = time.time()
    positions, velocities = simulate_swarm(
        num_oscillators=num_oscillators,
        num_steps=num_steps,
    )
    t_end = time.time()

    elapsed = t_end - t_start

    print(f"  Mean position : {np.mean(positions):>14.8f}")
    print(f"  Mean velocity : {np.mean(velocities):>14.8f}")
    print("-" * 60)
    print(f"  Total execution time : {elapsed:.4f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()
