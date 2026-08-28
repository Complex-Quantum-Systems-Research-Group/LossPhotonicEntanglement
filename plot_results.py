"""Generate all diagnostic and weak-coupling-check plots from an EP-MOKS
parameter sweep.

Usage:
    python plot_results.py [path/to/summary.json]

If no path is given, looks in config.output_root (default "data/") for the
file matching config.filename_tag(), falling back to the first
summary_*.json found there.

Outputs PNG figures to a "figs" directory, sister to the data directory
holding the summary file.

Section 1 -- basic diagnostics (from the sweep data alone):
  - concurrence, purity, Bell fidelity, mutual information, l1 coherence,
    relative entropy of coherence, and von Neumann entropy vs delay (fs),
    one curve per swept temperature (K)
  - degradation (concurrence / Bell fidelity at the longest simulated delay)
    vs temperature, with a marker at T_N = 39 K for KCuF3
  - recurrence detection for the finite chain (late-delay increases are
    physical and are reported, not mislabeled as monotonicity violations)

Section 2 -- independent check against the manuscript's second-order
weak-coupling prediction. For theta1 = theta2 = theta,

    rho_P(t) ~= [1 - 2 theta^2 D_M(t)] rho_Phi+ + 2 theta^2 D_M(t) rho_Psi-
                + O(theta^4),

where D_M(t) = (1/2) <[M(t) - M(0)]^2> is the mean-square temporal change of
the sampled magnetization in the thermal spin state, M(t) the
Heisenberg-picture magnetization operator, and <...> = Tr[rho_s(T) ...].
Since rho_Phi+ and rho_Psi- are orthogonal Bell states, this predicts

    Bell fidelity  F(t) ~= 1 - 2 theta^2 D_M(t) + O(theta^4)
    concurrence    C(t) ~= 1 - 4 theta^2 D_M(t) + O(theta^4)

D_M(t) is computed here directly from Hs, the thermal state, and the probe
magnetization operator -- all on the small 2^N-dimensional spin-only Hilbert
space (a single eigh call; the joint 4*2^N-dimensional operators are never
formed). It is NOT backed out from the simulated Bell fidelity/concurrence,
so overlaying it against the swept F(t)/C(t) is a genuine independent check
of the analytic theorem, not a circular one.

Closed form used (derived from the spectral decomposition of Hs; verified
against a brute-force Heisenberg-picture calculation to machine precision,
and against actual simulated data at theta=0.05 to O(theta^4) residuals,
before being used here):

    D_M(t) = sum_{a,b} p_a |M_ab|^2 [1 - cos((E_a - E_b) t)]

where |a> are Hs eigenstates, E_a its eigenvalues, p_a the thermal
populations, and M_ab = <a|M|b>. This form is manifestly non-negative for
all t, matching the manuscript's own claim that D_M(t) is "a single
non-negative quantity."
"""
from __future__ import annotations

import json
import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import config as cfg
from correlations import spectral_D_M, spectral_magnetization_correlators
from measures import bell_fidelity
from new_protocol import full_pipeline_unitary
from observables import partial_trace_spins
from states import bell_polarization_state


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def _load_summary(path=None):
    if path is None:
        data_dir = Path(cfg.output_root)
        expected = data_dir / f"summary_{cfg.filename_tag()}.json"
        if expected.exists():
            path = expected
        else:
            candidates = sorted(data_dir.glob("summary_*.json"))
            detail = f" Found {len(candidates)} legacy/config-mismatched summary file(s)." if candidates else ""
            raise FileNotFoundError(
                f"No summary matching the current configuration found at {expected}.{detail} "
                "Run 'python pipeline.py sweep', or pass a legacy summary explicitly: "
                "python plot_results.py path/to/summary.json"
            )
    else:
        path = Path(path)
    with open(path) as f:
        rows = json.load(f)
    return rows, path


def _figs_dir(summary_path, campaign=None):
    # figs is a sister directory of the data directory containing summary_path
    figs = summary_path.parent.parent / "figs"
    if campaign:
        figs = figs / campaign
    figs.mkdir(parents=True, exist_ok=True)
    return figs


def _group_by_temperature(rows, sort_key="delta_t_fs"):
    groups = {}
    for r in rows:
        groups.setdefault(r["T_kelvin"], []).append(r)
    for T in groups:
        groups[T] = sorted(groups[T], key=lambda r: r[sort_key])
    return groups


def _summary_metadata(rows, summary_path):
    """Read protocol parameters from rows, with legacy filename fallback."""
    first = rows[0]
    name = summary_path.stem

    def value(key, pattern, default, cast=float):
        if key in first:
            return cast(first[key])
        match = re.search(pattern, name)
        return cast(match.group(1)) if match else default

    return {
        "n_spins": value("n_spins", r"(?:^|_)N=([^_]+)", cfg.N_spins, int),
        "J": value("J", r"(?:^|_)J=([^_]+)", cfg.J),
        "delta": value("delta", r"(?:^|_)Delta=([^_]+)", cfg.delta),
        "h_z": value("h_z", r"(?:^|_)hz=([^_]+)", cfg.h_z),
        "theta1": value("theta1", r"(?:^|_)theta1=([^_]+)", cfg.theta1),
        "theta2": value("theta2", r"(?:^|_)theta2=([^_]+)", cfg.theta2),
        "probe_model": first.get("probe_model") or (
            re.search(r"(?:^|_)probe=(.+?)_sigma=", name).group(1)
            if re.search(r"(?:^|_)probe=(.+?)_sigma=", name) else cfg.probe_model
        ),
        "probe_sigma_sites": value(
            "probe_sigma_sites", r"(?:^|_)sigma=([^_]+)", cfg.probe_sigma_sites
        ),
        "periodic": bool(first.get("periodic", cfg.periodic)),
        "bell_state": first.get("bell_state", cfg.bell_state),
    }


# ---------------------------------------------------------------------
# Section 1: basic diagnostics
# ---------------------------------------------------------------------

def _temperature_delay_grid(rows, value):
    """Return rectangular temperature/delay coordinates and a value grid."""
    groups = _group_by_temperature(rows)
    temperatures = np.asarray(sorted(groups), dtype=float)
    delays = np.asarray([row["delta_t_fs"] for row in groups[temperatures[0]]])
    grid = np.asarray([
        [value(row) for row in groups[temperature]]
        for temperature in temperatures
    ], dtype=float)
    if grid.shape != (len(temperatures), len(delays)):
        raise ValueError("heatmaps require the same delay grid at every temperature")
    return temperatures, delays, grid


def plot_mixture_weight_heatmap(rows, figs_dir):
    """Plot the one independent output parameter p = 1 - <Phi+|rho|Phi+>."""
    temperatures, delays, mixture_weight = _temperature_delay_grid(
        rows, lambda row: 1.0 - row["bell_fidelity"],
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    mesh = ax.pcolormesh(delays, temperatures, mixture_weight, shading="auto", cmap="magma")
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label(r"mixture weight $p=1-F$")
    ax.set_xlabel("delay (fs)")
    ax.set_ylabel("temperature (K)")
    ax.set_title(
        rf"Exact Bell-mixture weight $p(\Delta t,T)$ "
        f"({cfg.material}, N={rows[0].get('n_spins', cfg.N_spins)})"
    )
    fig.tight_layout()
    fig.savefig(figs_dir / "mixture_weight_heatmap.png", dpi=180)
    plt.close(fig)


def plot_mixture_weight_cuts(rows, figs_dir):
    """Show readable delay cuts at low, middle, and high temperature."""
    groups = _group_by_temperature(rows)
    temperatures = sorted(groups)
    selected = sorted({temperatures[0], temperatures[len(temperatures) // 2], temperatures[-1]})
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for temperature in selected:
        group = groups[temperature]
        ax.plot(
            [row["delta_t_fs"] for row in group],
            [1.0 - row["bell_fidelity"] for row in group],
            label=f"T={temperature:.0f} K",
        )
    ax.set_xlabel("delay (fs)")
    ax.set_ylabel(r"mixture weight $p=1-F$")
    ax.set_title("Representative temperature cuts")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs_dir / "mixture_weight_selected_cuts.png", dpi=180)
    plt.close(fig)


def plot_degradation_vs_temperature(rows, figs_dir):
    groups = _group_by_temperature(rows)
    temps = sorted(groups.keys())
    minimum_C = [min(row["concurrence"] for row in groups[T]) for T in temps]
    minimum_F = [min(row["bell_fidelity"] for row in groups[T]) for T in temps]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(temps, minimum_C, marker="o", label=r"$\min_{\Delta t} C$")
    ax.plot(temps, minimum_F, marker="s", label=r"$\min_{\Delta t} F$")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("minimum over simulated delay")
    ax.set_title(f"Maximum degradation vs temperature ({cfg.material})")
    ax.axvline(39.0, color="gray", linestyle="--", alpha=0.6, label="$T_N$ = 39 K")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs_dir / "degradation_vs_temperature.png", dpi=150)
    plt.close(fig)


def characterize_extrema(rows, figs_dir):
    """Report physically meaningful extrema and recurrence spacings of p."""
    groups = _group_by_temperature(rows)
    records = []
    lines = ["Extrema of p=1-F (interior points; 5% prominence threshold):"]
    for temperature, group in groups.items():
        delays = np.asarray([row["delta_t_fs"] for row in group])
        p = 1.0 - np.asarray([row["bell_fidelity"] for row in group])
        prominence = max(1e-8, 0.05 * float(np.ptp(p)))
        p_maxima, _ = find_peaks(p, prominence=prominence)
        p_minima, _ = find_peaks(-p, prominence=prominence)
        fidelity_minima = delays[p_maxima]
        fidelity_maxima = delays[p_minima]
        periods = np.diff(fidelity_maxima)
        records.append({
            "T_kelvin": temperature,
            "prominence_threshold": prominence,
            "fidelity_minima_fs": ";".join(f"{value:.6g}" for value in fidelity_minima),
            "fidelity_maxima_fs": ";".join(f"{value:.6g}" for value in fidelity_maxima),
            "recurrence_periods_fs": ";".join(f"{value:.6g}" for value in periods),
        })
        minima_text = ", ".join(f"{value:.1f}" for value in fidelity_minima) or "none"
        maxima_text = ", ".join(f"{value:.1f}" for value in fidelity_maxima) or "none"
        period_text = ", ".join(f"{value:.1f}" for value in periods) or "n/a"
        lines.append(
            f"T={temperature:.0f} K: F minima [{minima_text}] fs; "
            f"F maxima [{maxima_text}] fs; periods [{period_text}] fs"
        )

    with (figs_dir / "extrema_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    fig, ax = plt.subplots(figsize=(10, 1.6 + 0.35 * len(records)))
    ax.axis("off")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=9, family="monospace")
    fig.tight_layout()
    fig.savefig(figs_dir / "extrema_recurrence_summary.png", dpi=180)
    plt.close(fig)
    return records


def run_basic_diagnostics(rows, figs_dir):
    plot_mixture_weight_heatmap(rows, figs_dir)
    plot_mixture_weight_cuts(rows, figs_dir)
    plot_degradation_vs_temperature(rows, figs_dir)

    extrema = characterize_extrema(rows, figs_dir)
    print(f"Characterized fidelity extrema at {len(extrema)} temperatures.")


# ---------------------------------------------------------------------
# Section 2: weak-coupling check
# ---------------------------------------------------------------------

def compute_D_M(n_spins, J, delta, temperature, delta_t_array,
                 probe_model, probe_sigma_sites, h_z=0.0, periodic=False):
    """D_M(t) at a single temperature, for an array of dimensionless delays.
    Cheap: a single eigh at the spin-only dimension (2^n_spins), then a
    vectorized sum per delay -- never touches the joint 4*2^n_spins space.
    """
    return spectral_D_M(
        n_spins, J, delta, temperature, delta_t_array,
        probe_model, probe_sigma_sites, h_z=h_z, periodic=periodic,
    )


def compute_magnetization_correlators(
    n_spins, J, delta, temperature, delta_t_array,
    probe_model, probe_sigma_sites, h_z=0.0, periodic=False,
):
    """Return both ordered correlators ``<M(t)M(0)>`` and ``<M(0)M(t)>``."""
    return spectral_magnetization_correlators(
        n_spins, J, delta, temperature, delta_t_array,
        probe_model, probe_sigma_sites, h_z=h_z, periodic=periodic,
    )


def export_correlators(rows, summary_path, metadata):
    """Export both operator orderings for every temperature and saved delay."""
    reports_dir = summary_path.parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / f"correlators_{summary_path.stem}.csv"
    records = []
    for temperature_kelvin, group in sorted(_group_by_temperature(rows, "delta_t").items()):
        delays = np.array([row["delta_t"] for row in group])
        c_t0, c_0t = compute_magnetization_correlators(
            metadata["n_spins"], metadata["J"], metadata["delta"], group[0]["T"], delays,
            metadata["probe_model"], metadata["probe_sigma_sites"],
            h_z=metadata["h_z"], periodic=metadata["periodic"],
        )
        for row, forward, reverse in zip(group, c_t0, c_0t):
            records.append({
                "T_kelvin": temperature_kelvin,
                "delta_t": row["delta_t"],
                "delta_t_fs": row["delta_t_fs"],
                "M_t_M_0_real": float(forward.real),
                "M_t_M_0_imag": float(forward.imag),
                "M_0_M_t_real": float(reverse.real),
                "M_0_M_t_imag": float(reverse.imag),
            })
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Exported both correlator orderings to {output_path}")


def fit_weak_coupling_exponent(metadata, groups):
    """Fit log|F_sim-F_pred| vs log(theta) at one representative point."""
    n_fit = min(metadata["n_spins"], 6)
    representative = groups[max(groups)]
    delays = np.array([row["delta_t"] for row in representative])
    d_m = compute_D_M(
        n_fit, metadata["J"], metadata["delta"], representative[0]["T"], delays,
        metadata["probe_model"], metadata["probe_sigma_sites"],
        h_z=metadata["h_z"], periodic=metadata["periodic"],
    )
    index = int(np.argmax(d_m))
    delay = float(delays[index])
    d_value = float(d_m[index])
    target = bell_polarization_state(metadata["bell_state"])
    theta_values = np.asarray(cfg.weak_coupling_theta_values, dtype=float)
    residuals = []
    for theta in theta_values:
        full = full_pipeline_unitary(
            n_spins=n_fit, J=metadata["J"], delta=metadata["delta"],
            temperature=representative[0]["T"], delta_t=delay,
            theta1=float(theta), theta2=float(theta),
            probe_model=metadata["probe_model"],
            probe_sigma_sites=metadata["probe_sigma_sites"],
            h_z=metadata["h_z"], periodic=metadata["periodic"],
            interaction_type="kerr", bell_state=metadata["bell_state"],
        )
        simulated = bell_fidelity(partial_trace_spins(full, n_fit), target)
        predicted = 1.0 - 2.0 * theta**2 * d_value
        residuals.append(abs(simulated - predicted))
    residuals = np.asarray(residuals)
    if np.any(residuals <= np.finfo(float).eps):
        raise AssertionError("weak-coupling exponent fit encountered a zero residual")
    exponent, intercept = np.polyfit(np.log(theta_values), np.log(residuals), 1)
    return {
        "exponent": float(exponent),
        "intercept": float(intercept),
        "n_spins": n_fit,
        "temperature_kelvin": float(max(groups)),
        "delta_t": delay,
        "theta_values": theta_values.tolist(),
        "residuals": residuals.tolist(),
    }


def plot_second_order_departure_heatmap(rows, figs_dir, metadata):
    """Plot relative departure of p from the equal-coupling second-order law."""
    theta1, theta2 = metadata["theta1"], metadata["theta2"]
    if not np.isclose(theta1, theta2):
        print("SKIP second-order departure heatmap: theta1 != theta2")
        return

    groups = _group_by_temperature(rows, sort_key="delta_t")
    temperatures = np.asarray(sorted(groups), dtype=float)
    delays_fs = np.asarray([row["delta_t_fs"] for row in groups[temperatures[0]]])
    relative_departure = []
    for temperature in temperatures:
        group = groups[temperature]
        delays = np.asarray([row["delta_t"] for row in group])
        # This is a descriptive production comparison, separate from the
        # weak-coupling validation gate exercised through compute_D_M below.
        d_m = spectral_D_M(
            metadata["n_spins"], metadata["J"], metadata["delta"], group[0]["T"], delays,
            metadata["probe_model"], metadata["probe_sigma_sites"],
            h_z=metadata["h_z"], periodic=metadata["periodic"],
        )
        predicted_p = 2.0 * theta1**2 * d_m
        simulated_p = 1.0 - np.asarray([row["bell_fidelity"] for row in group])
        relative_departure.append(np.divide(
            np.abs(simulated_p - predicted_p), predicted_p,
            out=np.full_like(predicted_p, np.nan), where=predicted_p > 1e-12,
        ))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    mesh = ax.pcolormesh(
        delays_fs, temperatures, np.asarray(relative_departure),
        shading="auto", cmap="viridis",
    )
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label(r"$|p-p_{\rm 2nd}|/p_{\rm 2nd}$")
    ax.set_xlabel("delay (fs)")
    ax.set_ylabel("temperature (K)")
    ax.set_title(
        rf"Relative departure from $p_{{\rm 2nd}}=2\theta^2D_M$ "
        f"({cfg.material}, theta={theta1:.3g}, N={metadata['n_spins']})"
    )
    fig.tight_layout()
    fig.savefig(figs_dir / "second_order_relative_departure_heatmap.png", dpi=180)
    plt.close(fig)


def run_weak_coupling_check(rows, figs_dir, summary_path):
    groups = _group_by_temperature(rows, sort_key="delta_t")
    metadata = _summary_metadata(rows, summary_path)
    theta1, theta2 = metadata["theta1"], metadata["theta2"]
    is_weak_campaign = (
        np.isclose(theta1, cfg.weak_coupling_theta)
        and np.isclose(theta2, cfg.weak_coupling_theta)
    )
    theta = theta1 if is_weak_campaign else None
    plot_second_order_departure_heatmap(rows, figs_dir, metadata)

    temps = sorted(groups.keys())
    cmap = plt.get_cmap("viridis")
    n_temps = max(len(temps) - 1, 1)

    fig_F, ax_F = plt.subplots(figsize=(7.5, 5.5))
    fig_C, ax_C = plt.subplots(figsize=(7.5, 5.5))
    fig_resid, ax_resid = (
        plt.subplots(figsize=(7.5, 5.5)) if is_weak_campaign else (None, None)
    )

    max_residual = 0.0
    for i, T_kelvin in enumerate(temps):
        rs = groups[T_kelvin]
        delta_t_dimensionless = np.array([r["delta_t"] for r in rs])
        delta_t_fs = np.array([r["delta_t_fs"] for r in rs])
        F_sim = np.array([r["bell_fidelity"] for r in rs])
        C_sim = np.array([r["concurrence"] for r in rs])
        T_dimensionless = rs[0]["T"]

        color = cmap(i / n_temps)

        ax_F.plot(delta_t_fs, F_sim, marker="o", markersize=3, color=color,
                  label=f"T={T_kelvin:.0f}K (sim)")
        ax_C.plot(delta_t_fs, C_sim, marker="o", markersize=3, color=color,
                  label=f"T={T_kelvin:.0f}K (sim)")

        if theta is not None:
            D_M = compute_D_M(
                metadata["n_spins"], metadata["J"], metadata["delta"], T_dimensionless,
                delta_t_dimensionless, metadata["probe_model"], metadata["probe_sigma_sites"],
                h_z=metadata["h_z"], periodic=metadata["periodic"],
            )
            F_pred = 1.0 - 2.0 * theta ** 2 * D_M
            C_pred = 1.0 - 4.0 * theta ** 2 * D_M

            ax_F.plot(delta_t_fs, F_pred, linestyle="--", color=color, alpha=0.7,
                      label=f"T={T_kelvin:.0f}K (2nd-order predicted)")
            ax_C.plot(delta_t_fs, C_pred, linestyle="--", color=color, alpha=0.7,
                      label=f"T={T_kelvin:.0f}K (2nd-order predicted)")

            resid = np.abs(F_sim - F_pred)
            max_residual = max(max_residual, resid.max())
            ax_resid.plot(delta_t_fs, resid, marker="o", markersize=3, color=color,
                          label=f"T={T_kelvin:.0f}K")

    ax_F.set_xlabel("delay (fs)")
    ax_F.set_ylabel("Bell fidelity")
    ax_F.set_title(
        f"Bell fidelity: "
        f"{'simulated vs 2nd-order prediction' if is_weak_campaign else 'simulated data'}\n"
        f"({cfg.material}, theta={theta1:.3g}, N={metadata['n_spins']})"
    )
    ax_F.legend(fontsize=6, ncol=2)
    ax_F.grid(alpha=0.3)
    fig_F.tight_layout()
    fidelity_filename = (
        "bell_fidelity_weak_coupling_check.png"
        if is_weak_campaign else "bell_fidelity_simulated.png"
    )
    fig_F.savefig(figs_dir / fidelity_filename, dpi=150)
    plt.close(fig_F)

    ax_C.set_xlabel("delay (fs)")
    ax_C.set_ylabel("Concurrence")
    ax_C.set_title(
        f"Concurrence: "
        f"{'simulated vs 2nd-order prediction' if is_weak_campaign else 'simulated data'}\n"
        f"({cfg.material}, theta={theta1:.3g}, N={metadata['n_spins']})"
    )
    ax_C.legend(fontsize=6, ncol=2)
    ax_C.grid(alpha=0.3)
    fig_C.tight_layout()
    concurrence_filename = (
        "concurrence_weak_coupling_check.png"
        if is_weak_campaign else "concurrence_simulated.png"
    )
    fig_C.savefig(figs_dir / concurrence_filename, dpi=150)
    plt.close(fig_C)

    if not is_weak_campaign:
        print(
            f"SKIP weak-coupling check: theta1={theta1:.3g}, theta2={theta2:.3g} "
            "is outside the perturbative regime (1% relative criterion holds "
            "for theta <~ 0.26). Plotting simulated data only."
        )
        export_correlators(rows, summary_path, metadata)
        return

    ax_resid.set_xlabel("delay (fs)")
    ax_resid.set_ylabel("|F_simulated - F_predicted|")
    ax_resid.set_yscale("log")
    ax_resid.set_title(
        f"Weak-coupling residual (equal-coupling Phi+ leading remainder is O(theta^4))\n"
        f"({cfg.material}, theta={theta1:.3g})"
    )
    ax_resid.legend(fontsize=6, ncol=2)
    ax_resid.grid(alpha=0.3, which="both")
    fig_resid.tight_layout()
    fig_resid.savefig(figs_dir / "weak_coupling_residual.png", dpi=150)
    plt.close(fig_resid)

    if theta is not None:
        relative_errors = []
        for rs in groups.values():
            delays = np.array([r["delta_t"] for r in rs])
            D_M = compute_D_M(
                metadata["n_spins"], metadata["J"], metadata["delta"], rs[0]["T"], delays,
                metadata["probe_model"], metadata["probe_sigma_sites"],
                h_z=metadata["h_z"], periodic=metadata["periodic"],
            )
            prediction = 1.0 - 2.0 * theta**2 * D_M
            signal = 1.0 - prediction
            simulated = np.array([r["bell_fidelity"] for r in rs])
            mask = signal > 1e-12
            relative_errors.extend((np.abs(simulated[mask] - prediction[mask]) / signal[mask]).tolist())
        max_relative_error = max(relative_errors, default=0.0)
        print(f"Max |F_sim - F_predicted| across sweep: {max_residual:.3e}")
        print(f"Max relative error / predicted signal: {max_relative_error:.3e}")
        exponent_fit = fit_weak_coupling_exponent(metadata, groups)
        exponent_pass = 3.5 <= exponent_fit["exponent"] <= 4.5
        print(
            f"Fitted log-residual exponent: {exponent_fit['exponent']:.4f} "
            f"(expected 4; N={exponent_fit['n_spins']}; "
            f"pass={exponent_pass})"
        )
        if max_relative_error >= 1e-2:
            print(
                "WARNING: second-order relative error exceeds 1%; this dataset "
                "is outside the strict weak-coupling validation regime."
            )
        else:
            print("Relative residual is below the 1% weak-coupling criterion.")
    export_correlators(rows, summary_path, metadata)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    rows, summary_path = _load_summary(path)
    print(f"Loaded {len(rows)} rows from {summary_path}")

    campaign = rows[0].get("campaign")
    figs_dir = _figs_dir(summary_path, campaign=campaign)
    print(f"Writing figures to {figs_dir}")

    print("\n--- Section 1: basic diagnostics ---")
    run_basic_diagnostics(rows, figs_dir)

    print("\n--- Section 2: weak-coupling check ---")
    run_weak_coupling_check(rows, figs_dir, summary_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
