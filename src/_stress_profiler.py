"""
Temporary stress-test & profiler helper for Exercise 3.

• Alpha (Stress Tester): sweeps sigma values for the log-normal cost
  distribution and records VaR95 for each.
• Beta (Profiler): runs the same JAX-jitted simulation twice **within
  this process** to compare cold (trace + XLA compile) vs warm (cached)
  execution time.

This script does NOT modify src/monte_carlo.py.
"""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

# ── Configuration ────────────────────────────────────────────────────
NUM_PATHS = 1_000_000


# =====================================================================
# Alpha – Stress Tester
# =====================================================================

def simulate_path_sigma(key, sigma):
    """Revenue path with a tuneable cost-volatility *sigma*."""
    key_d, key_c, key_r = jax.random.split(key, 3)

    D = jax.random.normal(key_d) * 150.0 + 1000.0
    C = jnp.exp(jax.random.normal(key_c) * sigma + 5.5)
    R = jax.random.uniform(key_r, minval=0.05, maxval=0.25)

    revenue = (D * 150.0) - C * (1.0 - R)
    return revenue


def run_stress_test():
    """Sweep sigma and return list of (sigma, var95, expected) tuples."""
    # Coarse sweep first, then refine around the break point
    sigmas_coarse = [0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]

    key = jax.random.PRNGKey(42)
    subkeys = jax.random.split(key, NUM_PATHS)

    results = []
    breaking_sigma = None

    print("── Alpha: Stress Test ──────────────────────────────────")
    print(f"  {'sigma':>6}  {'E[Revenue]':>14}  {'VaR95':>14}")
    print(f"  {'─'*6}  {'─'*14}  {'─'*14}")

    for s in sigmas_coarse:
        sim = jax.vmap(lambda k: simulate_path_sigma(k, s))
        revs = sim(subkeys).block_until_ready()
        expected = float(jnp.mean(revs))
        var95 = float(jnp.percentile(revs, 5))
        results.append((s, var95, expected))
        flag = " ← VaR95 < 0!" if var95 < 0 else ""
        print(f"  {s:6.2f}  {expected:>14,.2f}  {var95:>14,.2f}{flag}")

        if var95 < 0 and breaking_sigma is None:
            breaking_sigma = s

    # Fine-grained binary search around the breaking region
    if breaking_sigma is not None and len(results) >= 2:
        # Find the last positive VaR95 sigma
        positive_sigmas = [(s, v) for s, v, _ in results if v >= 0]
        if positive_sigmas:
            lo = positive_sigmas[-1][0]
            hi = breaking_sigma
            print(f"\n  Refining between sigma={lo:.2f} and sigma={hi:.2f} …")
            for _ in range(8):  # 8 bisection steps
                mid = (lo + hi) / 2.0
                sim = jax.vmap(lambda k: simulate_path_sigma(k, mid))
                revs = sim(subkeys).block_until_ready()
                var95 = float(jnp.percentile(revs, 5))
                expected = float(jnp.mean(revs))
                results.append((mid, var95, expected))
                print(f"  {mid:6.3f}  {expected:>14,.2f}  {var95:>14,.2f}")
                if var95 < 0:
                    hi = mid
                else:
                    lo = mid
            breaking_sigma = (lo + hi) / 2.0

    print()
    return results, breaking_sigma


# =====================================================================
# Beta – Profiler (in-process, same JIT cache)
# =====================================================================

def _simulate_path_baseline(key):
    """Baseline simulation (sigma=0.3) matching Exercise 2 logic."""
    return simulate_path_sigma(key, 0.3)


def run_profiler():
    """Run a jitted vmap simulation twice in-process; return (cold, warm, speedup)."""
    print("── Beta: Profiler (in-process) ─────────────────────────")

    key = jax.random.PRNGKey(42)
    subkeys = jax.random.split(key, NUM_PATHS)

    # Create a jit-compiled, vectorised simulation
    simulate_all = jax.jit(jax.vmap(_simulate_path_baseline))

    # Run 1 – cold: includes JAX tracing + XLA compilation
    t0 = time.perf_counter()
    rev1 = simulate_all(subkeys).block_until_ready()
    cold_time = time.perf_counter() - t0

    # Run 2 – warm: reuses the compiled HLO from the cache
    t0 = time.perf_counter()
    rev2 = simulate_all(subkeys).block_until_ready()
    warm_time = time.perf_counter() - t0

    speedup = cold_time / warm_time if warm_time > 0 else float("inf")

    print(f"  Run 1 [Cold – trace + compile]: {cold_time:.4f} s")
    print(f"  Run 2 [Warm – cached kernel ]:  {warm_time:.4f} s")
    print(f"  Speedup:                        {speedup:.1f}×")
    print()
    return cold_time, warm_time, speedup


# =====================================================================
# Main
# =====================================================================

def main():
    stress_results, breaking_sigma = run_stress_test()
    cold_time, warm_time, speedup = run_profiler()

    # Dump to JSON so the report generator can read structured data
    out = Path(__file__).resolve().parent.parent / "data" / "_stress_profiler_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stress": [{"sigma": s, "var95": v, "expected": e} for s, v, e in stress_results],
        "breaking_sigma": breaking_sigma,
        "profiler": {
            "cold_s": cold_time,
            "warm_s": warm_time,
            "speedup": speedup,
        },
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"Raw results → {out}")


if __name__ == "__main__":
    main()
