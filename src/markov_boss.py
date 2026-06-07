"""
Module Alpha: Markov Chain State Evolution with Black Swan Shock.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
OUTPUT_IMAGE = DATA_DIR / "markov_boss_states.png"

# Setup matrices
P_BASE = jnp.array([
    [0.85, 0.12, 0.03],
    [0.10, 0.75, 0.15],
    [0.05, 0.20, 0.75],
])

P_SHOCK = jnp.array([
    [0.10, 0.10, 0.80],
    [0.10, 0.10, 0.80],
    [0.05, 0.20, 0.75],
])

def transition_step(state, day):
    """
    Applies the transition matrix to the state vector for a given day.
    """
    # Check if the current day is within the Black Swan shock period (days 180-189 inclusive)
    is_shock = jnp.logical_and(day >= 180, day <= 189)
    P_current = jax.lax.select(is_shock, P_SHOCK, P_BASE)
    
    # Calculate the new state probability vector (state @ P_current)
    next_state = jnp.dot(state, P_current)
    
    return next_state, next_state

def main():
    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Initial state: 100% Bull Market
    initial_state = jnp.array([1.0, 0.0, 0.0])
    
    # 365 transition steps (days 0 to 364)
    days = jnp.arange(365)
    
    # jax.lax.scan runs the loop purely in JAX
    # Returns (final_state, history_of_states)
    _, states_history = jax.lax.scan(transition_step, initial_state, days)
    
    # Include the initial state at day 0
    all_states = jnp.vstack([initial_state, states_history])
    
    # Convert to percentages for the graph
    all_states_pct = all_states * 100.0

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    x_axis = jnp.arange(366)
    
    ax.plot(x_axis, all_states_pct[:, 0], label="Bull Market (State 0)", color="forestgreen", linewidth=2)
    ax.plot(x_axis, all_states_pct[:, 1], label="Stagnation (State 1)", color="goldenrod", linewidth=2)
    ax.plot(x_axis, all_states_pct[:, 2], label="Catastrophic Recession (State 2)", color="crimson", linewidth=2)
    
    # Highlight the Black Swan shock period
    ax.axvspan(180, 189, color='gray', alpha=0.3, label="Black Swan Shock (Days 180-189)")

    ax.set_title("Markov Chain Macro State Evolution (365 Days)")
    ax.set_xlabel("Day")
    ax.set_ylabel("Probability (%)")
    ax.set_xlim(0, 365)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    
    fig.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Graph successfully exported to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()
