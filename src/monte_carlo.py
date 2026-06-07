"""
Monte Carlo revenue simulation using JAX.

Simulates 1,000,000 paths in parallel with jax.vmap to estimate
expected revenue and Value-at-Risk (95 %) for a simple business model.
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


# ── Configuration ────────────────────────────────────────────────────
NUM_PATHS = 1_000_000
OUTPUT_IMAGE = Path(__file__).resolve().parent.parent / "data" / "revenue_dist.png"
SUBMISSION_LOG = Path(__file__).resolve().parent.parent / "docs" / "submission_log.md"


# ── Pure JAX simulation function ─────────────────────────────────────
def simulate_path(key):
    """Simulate a single revenue path.

    Parameters
    ----------
    key : jax.random.PRNGKey
        A unique PRNG key for this path.

    Returns
    -------
    revenue : scalar
        Net revenue for this path.
    """
    key_d, key_c, key_r = jax.random.split(key, 3)

    # Market Demand: D ~ Normal(mu=1000, sigma=150)
    D = jax.random.normal(key_d) * 150.0 + 1000.0

    # Production Asset Cost: ln(C) ~ Normal(mu=5.5, sigma=0.3)
    C = jnp.exp(jax.random.normal(key_c) * 0.3 + 5.5)

    # Regulatory Penalty Rate: R ~ Uniform(0.05, 0.25)
    R = jax.random.uniform(key_r, minval=0.05, maxval=0.25)

    # Net Revenue
    revenue = (D * 150.0) - C * (1.0 - R)
    return revenue


# ── Visualization ────────────────────────────────────────────────────
def save_histogram(
    revenues: np.ndarray,
    expected: float,
    var95: float,
    path: Path,
) -> None:
    """Plot the revenue distribution histogram and save to *path*."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(revenues, bins=200, color="steelblue", alpha=0.75, edgecolor="none")

    # Expected revenue (solid black)
    ax.axvline(expected, color="black", linewidth=1.8, label=f"Expected = {expected:,.0f}")

    # VaR95 (red dashed)
    ax.axvline(var95, color="red", linewidth=1.8, linestyle="--", label=f"VaR95 = {var95:,.0f}")

    ax.set_title("Monte Carlo Revenue Distribution  (1 000 000 paths)", fontsize=14)
    ax.set_xlabel("Net Revenue", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {path}")


# ── Submission log ───────────────────────────────────────────────────
def append_submission_log(
    elapsed: float,
    expected: float,
    var95: float,
    path: Path,
) -> None:
    """Append results to the Markdown submission log."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists() or path.stat().st_size == 0:
        header = (
            "# Submission Log\n\n"
            "| Method | Estimated π | Execution Time (s) | Points |\n"
            "|--------|-------------|---------------------|--------|\n"
        )
        path.write_text(header)

    row = (
        f"| JAX Monte Carlo Revenue | "
        f"E[Rev]={expected:,.2f}  VaR95={var95:,.2f} | "
        f"{elapsed:.4f} | "
        f"{NUM_PATHS:,} |\n"
    )
    with path.open("a") as f:
        f.write(row)

    print(f"Result appended → {path}")


# ── Main ─────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Running JAX Monte Carlo revenue simulation with {NUM_PATHS:,} paths …\n")

    # Master key + subkeys
    key = jax.random.PRNGKey(42)
    subkeys = jax.random.split(key, NUM_PATHS)

    # Vectorised simulation
    simulate_all = jax.vmap(simulate_path)

    t0 = time.perf_counter()
    revenues = simulate_all(subkeys).block_until_ready()
    elapsed = time.perf_counter() - t0

    # Statistics
    expected = float(jnp.mean(revenues))
    var95 = float(jnp.percentile(revenues, 5))

    print(f"  Execution time   = {elapsed:.4f} s")
    print(f"  Expected revenue = {expected:,.2f}")
    print(f"  VaR95 threshold  = {var95:,.2f}\n")

    # Convert to NumPy for matplotlib
    revenues_np = np.asarray(revenues)

    save_histogram(revenues_np, expected, var95, OUTPUT_IMAGE)
    append_submission_log(elapsed, expected, var95, SUBMISSION_LOG)


if __name__ == "__main__":
    main()
