"""Finite-size scaling campaign for the exact spin-only D_M correlator."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

import config as cfg
from correlations import spectral_D_M
from sector_correlations import infinite_temperature_M2
from sector_correlations import diagonalize_magnetization_sectors, sector_D_M


def extrema(delays, values):
    prominence = max(1e-12, 0.05 * float(np.ptp(values)))
    maxima, _ = find_peaks(values, prominence=prominence)
    minima, _ = find_peaks(-values, prominence=prominence)
    return delays[maxima], delays[minima], prominence


def production_comparison(delays, dm, temperature):
    paths = sorted(Path(cfg.output_root).glob("summary*theta1=0.4_theta2=0.4_int=kerr.json"))
    paths = [p for p in paths if "campaign=" not in p.name]
    if not paths:
        return None
    rows = json.loads(paths[0].read_text(encoding="utf-8"))
    group = sorted(
        (r for r in rows if np.isclose(r["T"], temperature)),
        key=lambda r: r["delta_t"],
    )
    if not group:
        return None
    grid = np.asarray([r["delta_t"] for r in group])
    p = 1.0 - np.asarray([r["bell_fidelity"] for r in group])
    mapped = np.interp(grid, delays, dm)
    predicted = 2.0 * cfg.theta1**2 * mapped
    mask = predicted > 1e-12
    departure = np.abs(p[mask] - predicted[mask]) / predicted[mask]
    pmax, pmin, _ = extrema(grid, p)
    dmax, dmin, _ = extrema(delays, dm)
    return {
        "relative_departure_min": float(departure.min()),
        "relative_departure_max": float(departure.max()),
        "p_maxima": pmax.tolist(), "p_minima": pmin.tolist(),
        "dm_maxima": dmax.tolist(), "dm_minima": dmin.tolist(),
        "nearest_first_maximum_error": float(np.min(np.abs(dmax - pmax[0]))) if len(pmax) and len(dmax) else None,
        "nearest_first_revival_error": float(np.min(np.abs(dmin - pmin[0]))) if len(pmin) and len(dmin) else None,
    }


def validate_n10(temperatures):
    delays = np.asarray(cfg.delta_t_list, dtype=float)
    half, weights = diagonalize_magnetization_sectors(
        10, cfg.J, cfg.delta, cfg.probe_model, cfg.probe_sigma_sites,
        h_z=0.0, periodic=cfg.periodic, use_spin_flip=True,
    )
    full, _ = diagonalize_magnetization_sectors(
        10, cfg.J, cfg.delta, cfg.probe_model, cfg.probe_sigma_sites,
        h_z=0.0, periodic=cfg.periodic, use_spin_flip=False,
    )
    worst_reference = worst_parity = 0.0
    for temperature in temperatures:
        dm_half, _ = sector_D_M(half, temperature, delays, relative_cutoff=0.0)
        dm_full, _ = sector_D_M(full, temperature, delays, relative_cutoff=0.0)
        reference = spectral_D_M(
            10, cfg.J, cfg.delta, temperature, delays,
            cfg.probe_model, cfg.probe_sigma_sites, h_z=0.0, periodic=cfg.periodic,
        )
        worst_parity = max(worst_parity, float(np.max(np.abs(dm_half - dm_full))))
        worst_reference = max(worst_reference, float(np.max(np.abs(dm_half - reference))))
    _, high = sector_D_M(half, 1e12, [0.0], relative_cutoff=0.0)
    infinity_error = abs(high["S0"] - infinite_temperature_M2(weights))
    if max(worst_parity, worst_reference, infinity_error) >= 1e-12:
        raise AssertionError(
            f"N=10 sector validation failed: parity={worst_parity}, "
            f"reference={worst_reference}, Tinf={infinity_error}"
        )
    print(
        f"[PASS] N=10 pointwise reference={worst_reference:.3e}, "
        f"parity pairing={worst_parity:.3e}, Tinf={infinity_error:.3e}"
    )


NS = np.array([8, 10, 12, 14])
FIRST_MAX_6 = np.array([2.4, 2.2, 2.4, 2.2])
FIRST_REV_6 = np.array([4.3, 3.8, 4.4, 4.1])
FIRST_MAX_500 = np.array([5.6, 7.7, 10.6, 13.1])


def plot_reference():
    """Plot the original embedded reference arrays, not new campaign results."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.4, 2.8), constrained_layout=True)
    ax.plot(NS, FIRST_MAX_6, "o-", label=r"first max, 6 K")
    ax.plot(NS, FIRST_REV_6, "s-", label=r"first revival, 6 K")
    ax.plot(NS, FIRST_MAX_500, "^-", label=r"first max, 500 K")

    n = np.linspace(7, 15, 2)
    ax.plot(n, 2 * n / np.pi, "k--", lw=1, label=r"$2N/\pi J$ (traversal)")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"extremum position ($\hbar/J$)")
    ax.set_xticks(NS)
    ax.legend(fontsize=6, frameon=False)

    output = Path("img/finite_size_scaling.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"wrote {output}")


def report_revivals(curves_path):
    """Report first/second revivals for the original reference campaign."""
    t_keys = {6: "0.0152070587", 500: "1.267254891"}
    with np.load(curves_path) as d:
        for T in (6, 500):
            print(f"\n--- T = {T} K, DM minima (revivals), hbar/J ---")
            first, second = [], []
            for N in NS:
                minima = extrema(d[f"delays_N{N}"], d[f"N{N}_T{t_keys[T]}"])[1]
                print(f"  N={N:2d}: {np.round(minima, 2)}")
                first.append(minima[0] if len(minima) > 0 else np.nan)
                second.append(minima[1] if len(minima) > 1 else np.nan)
            for label, vals in (("first", first), ("second", second)):
                values = np.array(vals, float)
                ok = ~np.isnan(values)
                if ok.sum() < 3:
                    print(f"  {label} revival: fewer than 3 resolved, no fit")
                    continue
                slope, intercept = np.polyfit(np.array(NS)[ok], values[ok], 1)
                print(
                    f"  {label} revival: slope={slope:+.3f}/J  "
                    f"intercept={intercept:+.2f}  | one-way 2/pi={2/np.pi:.3f}  "
                    f"round-trip 4/pi={4/np.pi:.3f}"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[8, 10, 12, 14])
    parser.add_argument("--temperatures", nargs="+", type=float,
                        default=[float(cfg.temperature_list[0]), float(cfg.temperature_list[-1])])
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--cutoff", type=float, default=1e-14)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--plot-reference", action="store_true",
                         help="Plot the original embedded finite-size reference arrays")
    actions.add_argument("--revivals", action="store_true",
                         help="Analyze first/second revivals in saved reference curves")
    parser.add_argument("--curves", type=Path, default=Path("reports/finite_size_dm_curves.npz"))
    args = parser.parse_args()
    if args.plot_reference:
        plot_reference()
        return
    if args.revivals:
        report_revivals(args.curves)
        return

    outdir = Path("reports")
    outdir.mkdir(exist_ok=True)
    validate_n10(args.temperatures)
    records, curves = [], {}
    for n in args.sizes:
        t_trav = 2.0 * n / (np.pi * cfg.J)
        delays = np.arange(0.0, 2.5 * t_trav + 0.5 * args.step, args.step)
        print(f"N={n}: diagonalizing sectors through k={n//2} ...", flush=True)
        sectors, _ = diagonalize_magnetization_sectors(
            n, cfg.J, cfg.delta, cfg.probe_model, cfg.probe_sigma_sites,
            h_z=0.0, periodic=cfg.periodic, use_spin_flip=True,
        )
        for temperature in args.temperatures:
            print(f"N={n}, T={temperature:.7g}: evaluating {len(delays)} delays ...", flush=True)
            dm, meta = sector_D_M(sectors, temperature, delays, args.cutoff)
            if dm[0] != 0.0:
                raise AssertionError(f"D_M(0) must be exact zero, got {dm[0]} at N={n}")
            maxima, minima, prominence = extrema(delays, dm)
            record = {
                "N": n, "T": temperature, "T_kelvin": float(cfg.temperature_kelvin(temperature)),
                "t_trav": t_trav, "window_max": float(delays[-1]), "step": args.step,
                "prominence": prominence,
                "first_maximum": float(maxima[0]) if len(maxima) else None,
                "first_revival": float(minima[0]) if len(minima) else None,
                "maxima": maxima.tolist(), "minima": minima.tolist(), **meta,
            }
            if n == cfg.N_spins:
                record["production_comparison"] = production_comparison(delays, dm, temperature)
            records.append(record)
            curves[f"N{n}_T{temperature:.10g}"] = dm
            curves[f"delays_N{n}"] = delays

    fits = {}
    for temperature in args.temperatures:
        rs = [r for r in records if np.isclose(r["T"], temperature)]
        for field in ("first_maximum", "first_revival"):
            usable = [r for r in rs if r[field] is not None]
            if len(usable) >= 2:
                slope, intercept = np.polyfit([r["N"] for r in usable], [r[field] for r in usable], 1)
                fits[f"T={temperature:.10g}:{field}"] = {
                    "slope": float(slope), "intercept": float(intercept),
                    "traversal_slope": float(2.0 / (np.pi * cfg.J)),
                }

    report = {"parameters": {**vars(args), "curves": str(args.curves)}, "records": records, "fits": fits}
    (outdir / "finite_size_dm.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(outdir / "finite_size_dm_curves.npz", **curves)
    flat = [{k: v for k, v in r.items() if k not in {"maxima", "minima", "production_comparison"}} for r in records]
    with (outdir / "finite_size_dm.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader(); writer.writerows(flat)
    print(json.dumps(fits, indent=2))


if __name__ == "__main__":
    main()
