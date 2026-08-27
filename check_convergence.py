"""Convergence checks required by README.md before interpreting a sweep:
larger N_spins and alternative probe profiles, at a small set of
representative points rather than a full grid (both because a full N=12
grid is prohibitively expensive -- see module docstring below -- and
because a handful of well-chosen points is enough to detect drift).

This script deliberately reuses, rather than reimplements, the point
computation and I/O logic already validated elsewhere:
  - pipeline.run_point (parameterized to accept n_spins/probe_model/
    probe_sigma_sites overrides) computes every point here, so a point
    computed for a convergence check gets the exact same diagnostics,
    density-matrix validation, and real-unit conversions as a production
    sweep point -- there is no separate, parallel implementation that
    could silently drift out of sync.
  - plot_results._load_summary / _figs_dir locate the existing sweep
    output and the sister figs/ directory, matching where
    plot_results.py itself writes.

Usage
-----
Cheap check first (same N=10, no extra compute cost per point):
    python check_convergence.py probe-profile

Expensive check (larger N; TIME ONE POINT FIRST, see time-one-point below):
    python check_convergence.py time-one-point --N 12
    python check_convergence.py n-convergence --N 11
    python check_convergence.py n-convergence --N 12

Both n-convergence and probe-profile write a comparison figure and a text
summary to figs/ (sister of config.output_root), and print a summary table
to stdout.

Cost warning for n-convergence
-------------------------------
The fast evolution paths (new_evolution.py) scale with the spin-only
Hilbert space dimension 2^N_spins, and the dominant apply_kron_sum step
scales roughly cubically with it. Extrapolating from the measured N=10 cost
(52.5s/point):

    N=11 (dim 2048): ~8x  -> ~7 min/point,  ~4-6 GB peak memory
    N=12 (dim 4096): ~64x -> ~56 min/point, ~16-20 GB peak memory

These are extrapolations from cubic scaling, not measurements at N=11/12.
Actual scaling can differ (cache effects, BLAS threading). Run
`time-one-point` first and compare against this estimate before committing
to a multi-point comparison batch. Also confirm your instance has enough
RAM for the N=12 peak memory estimate before running it -- a memory error
partway through a long-running point wastes the whole point's compute.

N=12 is recommended over N=11 as the primary convergence check despite the
higher cost: N=10 and N=12 are both even-length open chains, so the
comparison avoids the differing ground-state degeneracy structure that an
odd-length open chain (N=11) introduces. N=11 is offered as a cheaper first
look, not a substitute.
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as cfg
import pipeline
from plot_results import _load_summary, _figs_dir


def _save_comparison_results(name, results, summary_path, metadata):
    """Persist comparison numbers so conclusions do not depend on a PNG."""
    reports_dir = summary_path.parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / f"{name}.csv"
    json_path = reports_dir / f"{name}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "rows": results}, handle, indent=2)
        handle.write("\n")
    print(f"Saved numeric comparison to {csv_path} and {json_path}")


def _pick_representative_points(rows, n_temps=2, n_delays=4):
    """Pick a small bracketing set of (T, delta_t) points: min and max
    temperature, crossed with delays spanning 0, the observed late-delay
    recurrence region, and the max simulated delay -- using only delta_t
    values that actually appear in the existing sweep, so the comparison
    is against real N=10 data, not an interpolated point.
    """
    all_T = sorted(set(r["T"] for r in rows))
    all_dt = sorted(set(r["delta_t"] for r in rows))
    chosen_T = [all_T[0], all_T[-1]] if n_temps == 2 else all_T[:n_temps]

    # spread n_delays indices evenly across the available delta_t grid,
    # always including the first and last
    idx = np.linspace(0, len(all_dt) - 1, n_delays).round().astype(int)
    idx = sorted(set(idx.tolist()))
    chosen_dt = [all_dt[i] for i in idx]

    points = []
    lookup = {(r["T"], r["delta_t"]): r for r in rows}
    for T in chosen_T:
        for dt in chosen_dt:
            key = (T, dt)
            if key in lookup:
                points.append(lookup[key])
    return points


def cmd_time_one_point(args):
    T = cfg.temperature_list[0]
    dt = cfg.delta_t_list[len(cfg.delta_t_list) // 2]
    print(f"Timing one point at N={args.N}, T={T:.4f}, delta_t={dt:.4f} ...")
    t0 = time.time()
    result = pipeline.run_point(T, dt, n_spins=args.N)
    elapsed = time.time() - t0
    print(f"Elapsed: {elapsed:.1f} s")
    print(
        f"concurrence={result['concurrence']:.6f}  "
        f"bell_fidelity={result['bell_fidelity']:.6f}"
    )
    n_points = 8
    print(
        f"At this rate, an {n_points}-point comparison batch would take "
        f"~{elapsed * n_points / 60:.1f} minutes."
    )


def cmd_n_convergence(args):
    rows, summary_path = _load_summary()
    figs_dir = _figs_dir(summary_path, campaign=rows[0].get("campaign"))
    points = _pick_representative_points(rows, n_temps=args.n_temps, n_delays=args.n_delays)
    print(f"Comparing {len(points)} representative points: N=10 (existing) vs N={args.N} (new)")

    results = []
    for i, p in enumerate(points):
        print(
            f"[{i+1}/{len(points)}] T={p['T_kelvin']:.0f}K, "
            f"delta_t={p['delta_t_fs']:.1f}fs ...", end=" ", flush=True,
        )
        t0 = time.time()
        new = pipeline.run_point(p["T"], p["delta_t"], n_spins=args.N)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")
        results.append({
            "T_kelvin": p["T_kelvin"],
            "delta_t_fs": p["delta_t_fs"],
            "N10_concurrence": p["concurrence"],
            "Nnew_concurrence": new["concurrence"],
            "N10_bell_fidelity": p["bell_fidelity"],
            "Nnew_bell_fidelity": new["bell_fidelity"],
            "concurrence_diff": abs(p["concurrence"] - new["concurrence"]),
            "bell_fidelity_diff": abs(p["bell_fidelity"] - new["bell_fidelity"]),
        })

    print(f"\n{'T(K)':>6} {'dt(fs)':>8} {'C_N10':>10} {'C_N' + str(args.N):>10} {'|diff|':>10} {'F_N10':>10} {'F_N' + str(args.N):>10} {'|diff|':>10}")
    for r in results:
        print(
            f"{r['T_kelvin']:6.0f} {r['delta_t_fs']:8.1f} "
            f"{r['N10_concurrence']:10.6f} {r['Nnew_concurrence']:10.6f} {r['concurrence_diff']:10.2e} "
            f"{r['N10_bell_fidelity']:10.6f} {r['Nnew_bell_fidelity']:10.6f} {r['bell_fidelity_diff']:10.2e}"
        )

    max_c_diff = max(r["concurrence_diff"] for r in results)
    max_f_diff = max(r["bell_fidelity_diff"] for r in results)
    print(f"\nMax |concurrence diff| = {max_c_diff:.3e}")
    print(f"Max |bell_fidelity diff| = {max_f_diff:.3e}")
    _save_comparison_results(
        f"n_convergence_N10_vs_N{args.N}", results, summary_path,
        {
            "baseline_N": cfg.N_spins,
            "comparison_N": args.N,
            "max_concurrence_diff": max_c_diff,
            "max_bell_fidelity_diff": max_f_diff,
        },
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(results))
    labels = [f"T={r['T_kelvin']:.0f}K\ndt={r['delta_t_fs']:.0f}fs" for r in results]
    axes[0].plot(x, [r["N10_concurrence"] for r in results], "o-", label="N=10")
    axes[0].plot(x, [r["Nnew_concurrence"] for r in results], "s--", label=f"N={args.N}")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    axes[0].set_ylabel("Concurrence")
    axes[0].set_title("Concurrence: N=10 vs N=" + str(args.N))
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, [r["N10_bell_fidelity"] for r in results], "o-", label="N=10")
    axes[1].plot(x, [r["Nnew_bell_fidelity"] for r in results], "s--", label=f"N={args.N}")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    axes[1].set_ylabel("Bell fidelity")
    axes[1].set_title("Bell fidelity: N=10 vs N=" + str(args.N))
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    outpath = figs_dir / f"n_convergence_N10_vs_N{args.N}.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"\nSaved comparison figure to {outpath}")


def cmd_probe_profile(args):
    rows, summary_path = _load_summary()
    figs_dir = _figs_dir(summary_path, campaign=rows[0].get("campaign"))
    points = _pick_representative_points(rows, n_temps=args.n_temps, n_delays=args.n_delays)

    alt_profiles = [
        ("local_gaussian", 0.5),
        ("local_gaussian", 2.0),
        ("single_site", 1.0),
    ]
    print(
        f"Comparing {len(points)} representative points against default "
        f"probe (probe_model={cfg.probe_model!r}, sigma={cfg.probe_sigma_sites}) "
        f"for {len(alt_profiles)} alternative profiles"
    )

    all_results = []
    for probe_model, sigma in alt_profiles:
        for p in points:
            new = pipeline.run_point(
                p["T"], p["delta_t"],
                probe_model=probe_model, probe_sigma_sites=sigma,
            )
            all_results.append({
                "profile": f"{probe_model}(sigma={sigma})" if probe_model == "local_gaussian" else probe_model,
                "T_kelvin": p["T_kelvin"],
                "delta_t_fs": p["delta_t_fs"],
                "default_concurrence": p["concurrence"],
                "alt_concurrence": new["concurrence"],
                "concurrence_diff": abs(p["concurrence"] - new["concurrence"]),
            })

    print(f"\n{'profile':>25} {'T(K)':>6} {'dt(fs)':>8} {'C_default':>12} {'C_alt':>10} {'|diff|':>10}")
    for r in all_results:
        print(
            f"{r['profile']:>25} {r['T_kelvin']:6.0f} {r['delta_t_fs']:8.1f} "
            f"{r['default_concurrence']:12.6f} {r['alt_concurrence']:10.6f} {r['concurrence_diff']:10.2e}"
        )

    max_diff = max(r["concurrence_diff"] for r in all_results)
    _save_comparison_results(
        "probe_profile_comparison", all_results, summary_path,
        {
            "default_probe_model": cfg.probe_model,
            "default_probe_sigma_sites": cfg.probe_sigma_sites,
            "max_concurrence_diff": max_diff,
        },
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    profiles = sorted(set(r["profile"] for r in all_results))
    for prof in profiles:
        rs = [r for r in all_results if r["profile"] == prof]
        x = [r["delta_t_fs"] for r in rs]
        y = [r["alt_concurrence"] for r in rs]
        ax.plot(x, y, marker="o", markersize=4, label=prof, alpha=0.8)
    ax.plot(
        [p["delta_t_fs"] for p in points], [p["concurrence"] for p in points],
        marker="s", markersize=6, color="black", linestyle="--",
        label=f"default ({cfg.probe_model}, sigma={cfg.probe_sigma_sites})",
    )
    ax.set_xlabel("delay (fs)")
    ax.set_ylabel("Concurrence")
    ax.set_title("Concurrence across probe profiles (representative points)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    outpath = figs_dir / "probe_profile_comparison.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"\nSaved comparison figure to {outpath}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_time = sub.add_parser("time-one-point", help="Time a single point at a given N before committing to a full comparison")
    p_time.add_argument("--N", type=int, required=True, help="N_spins to time (e.g. 11 or 12)")
    p_time.set_defaults(func=cmd_time_one_point)

    p_conv = sub.add_parser("n-convergence", help="Compare representative points between N=10 (existing sweep) and a larger N")
    p_conv.add_argument("--N", type=int, required=True, help="Larger N_spins to compare against N=10 (e.g. 11 or 12)")
    p_conv.add_argument("--n-temps", type=int, default=2, help="Number of temperatures to check (default: 2, min and max)")
    p_conv.add_argument("--n-delays", type=int, default=4, help="Number of delays to check (default: 4, spread across the grid)")
    p_conv.set_defaults(func=cmd_n_convergence)

    p_probe = sub.add_parser("probe-profile", help="Compare representative points against alternative probe profiles (cheap: same N=10)")
    p_probe.add_argument("--n-temps", type=int, default=2)
    p_probe.add_argument("--n-delays", type=int, default=4)
    p_probe.set_defaults(func=cmd_probe_profile)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
