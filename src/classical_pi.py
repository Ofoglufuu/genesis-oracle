"""
Classical Monte Carlo estimation of π using NumPy.

Generates 5,000,000 random (x, y) points in the unit square,
counts how many fall inside the quarter-circle of radius 1,
and estimates π ≈ 4 × (inside / total).
"""

import os
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ── Configuration ────────────────────────────────────────────────────
TOTAL_POINTS = 5_000_000
DISPLAY_POINTS = 10_000
OUTPUT_IMAGE = Path(__file__).resolve().parent.parent / "data" / "classical_pi_disp.png"
SUBMISSION_LOG = Path(__file__).resolve().parent.parent / "docs" / "submission_log.md"


def estimate_pi(n: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Return (pi_estimate, x, y, distance) using *n* random points."""
    x = np.random.uniform(0.0, 1.0, size=n)
    y = np.random.uniform(0.0, 1.0, size=n)
    distance = np.sqrt(x**2 + y**2)
    inside_count = np.sum(distance <= 1.0)
    pi_est = 4.0 * inside_count / n
    return pi_est, x, y, distance


def save_scatter(
    x: np.ndarray,
    y: np.ndarray,
    distance: np.ndarray,
    pi_est: float,
    path: Path,
) -> None:
    """Plot a 10 000-point sample and save to *path*."""
    # Randomly choose DISPLAY_POINTS indices
    rng = np.random.default_rng(42)
    idx = rng.choice(len(x), size=DISPLAY_POINTS, replace=False)
    xs, ys, ds = x[idx], y[idx], distance[idx]

    inside = ds <= 1.0
    outside = ~inside

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter – inside (blue) then outside (red)
    ax.scatter(xs[inside], ys[inside], s=1, c="royalblue", alpha=0.6, label="Inside")
    ax.scatter(xs[outside], ys[outside], s=1, c="crimson", alpha=0.6, label="Outside")

    # Quarter-circle boundary
    theta = np.linspace(0, np.pi / 2, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1.5, label="Quarter circle")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(f"Monte Carlo π ≈ {pi_est:.6f}  (10 000 of {TOTAL_POINTS:,} points shown)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="lower left", markerscale=8)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {path}")


def append_submission_log(pi_est: float, elapsed: float, path: Path) -> None:
    """Append a result row to the submission log (Markdown table)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # If the file doesn't exist yet, write the header first
    if not path.exists() or path.stat().st_size == 0:
        header = (
            "# Submission Log\n\n"
            "| Method | Estimated π | Execution Time (s) | Points |\n"
            "|--------|-------------|---------------------|--------|\n"
        )
        path.write_text(header)

    row = f"| Classical Monte Carlo | {pi_est:.10f} | {elapsed:.4f} | {TOTAL_POINTS:,} |\n"
    with path.open("a") as f:
        f.write(row)

    print(f"Result appended → {path}")


# ── Main ─────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Running Monte Carlo π estimation with {TOTAL_POINTS:,} points …\n")

    t0 = time.perf_counter()
    pi_est, x, y, distance = estimate_pi(TOTAL_POINTS)
    elapsed = time.perf_counter() - t0

    print(f"  Estimated π  = {pi_est:.10f}")
    print(f"  Execution time = {elapsed:.4f} s\n")

    save_scatter(x, y, distance, pi_est, OUTPUT_IMAGE)
    append_submission_log(pi_est, elapsed, SUBMISSION_LOG)


if __name__ == "__main__":
    main()
