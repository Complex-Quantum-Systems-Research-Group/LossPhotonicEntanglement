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
  - a monotonicity sanity check: concurrence/purity/fidelity should not
    increase with delay (up to numerical noise); violations are flagged,
    not silently ignored

Section 2 -- independent check against the manuscript's second-order
weak-coupling prediction. For theta1 = theta2 = theta,

    rho_P(t) ~= [1 - 2 theta^2 D_M(t)] rho_Phi+ + 2 theta^2 D_M(t) rho_Psi-
                + O(theta^3),

where D_M(t) = (1/2) <[M(t) - M(0)]^2> is the mean-square temporal change of
the sampled magnetization in the thermal spin state, M(t) the
Heisenberg-picture magnetization operator, and <...> = Tr[rho_s(T) ...].
Since rho_Phi+ and rho_Psi- are orthogonal Bell states, this predicts

    Bell fidelity  F(t) ~= 1 - 2 theta^2 D_M(t) + O(theta^3)
    concurrence    C(t) ~= 1 - 4 theta^2 D_M(t) + O(theta^3)

D_M(t) is computed here directly from Hs, the thermal state, and the probe
magnetization operator -- all on the small 2^N-dimensional spin-only Hilbert
space (a single eigh call; the joint 4*2^N-dimensional operators are never
formed). It is NOT backed out from the simulated Bell fidelity/concurrence,
so overlaying it against the swept F(t)/C(t) is a genuine independent check
of the analytic theorem, not a circular one.

Closed form used (derived from the spectral decomposition of Hs; verified
against a brute-force Heisenberg-picture calculation to machine precision,
and against actual simulated data at theta=0.05 to O(theta^3) residuals,
before being used here):

    D_M(t) = sum_{a,b} p_a |M_ab|^2 [1 - cos((E_a - E_b) t)]

where |a> are Hs eigenstates, E_a its eigenvalues, p_a the thermal
populations, and M_ab = <a|M|b>. This form is manifestly non-negative for
all t, matching the manuscript's own claim that D_M(t) is "a single
non-negative quantity."
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

import config as cfg
from hamiltonians import build_spin_hamiltonian_xxz, weighted_magnetization_z
from new_protocol import build_probe_weights


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
            if not candidates:
                raise FileNotFoundError(
                    f"No summary_*.json found in {data_dir}. "
                    "Run 'python pipeline.py sweep' first, or pass a path "
                    "explicitly: python plot_results.py path/to/summary.json"
                )
            path = candidates[0]
    else:
        path = Path(path)
    with open(path) as f:
        rows = json.load(f)
    return rows, path


def _figs_dir(summary_path):
    # figs is a sister directory of the data directory containing summary_path
    figs = summary_path.parent.parent / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    return figs


def _group_by_temperature(rows, sort_key="delta_t_fs"):
    groups = {}
    for r in rows:
        groups.setdefault(r["T_kelvin"], []).append(r)
    for T in groups:
        groups[T] = sorted(groups[T], key=lambda r: r[sort_key])
    return groups


# ---------------------------------------------------------------------
# Section 1: basic diagnostics
# ---------------------------------------------------------------------

def plot_metric_vs_delay(rows, metric, ylabel, figs_dir, filename, ylim=None):
    groups = _group_by_temperature(rows)
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("viridis")
    temps = sorted(groups.keys())
    n_temps = max(len(temps) - 1, 1)
    for i, T in enumerate(temps):
        rs = groups[T]
        x = [r["delta_t_fs"] for r in rs]
        y = [r[metric] for r in rs]
        ax.plot(
            x, y, marker="o", markersize=3,
            color=cmap(i / n_temps), label=f"T={T:.0f} K",
        )
    ax.set_xlabel("delay (fs)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs delay ({cfg.material}, N={cfg.N_spins})")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs_dir / filename, dpi=150)
    plt.close(fig)


def plot_degradation_vs_temperature(rows, figs_dir):
    groups = _group_by_temperature(rows)
    temps = sorted(groups.keys())
    max_delay_C = [groups[T][-1]["concurrence"] for T in temps]
    max_delay_F = [groups[T][-1]["bell_fidelity"] for T in temps]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(temps, max_delay_C, marker="o", label="concurrence at max delay")
    ax.plot(temps, max_delay_F, marker="s", label="Bell fidelity at max delay")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("value at longest simulated delay")
    ax.set_title(f"Degradation vs temperature ({cfg.material})")
    ax.axvline(39.0, color="gray", linestyle="--", alpha=0.6, label="$T_N$ = 39 K")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs_dir / "degradation_vs_temperature.png", dpi=150)
    plt.close(fig)


def check_monotonicity(rows, figs_dir, tol=1e-6):
    """Concurrence/purity/Bell fidelity should be non-increasing with delay
    (up to numerical noise) at fixed temperature. Writes a text-summary
    figure and returns the list of violations found (empty if none).
    """
    groups = _group_by_temperature(rows)
    violations = []
    for T, rs in groups.items():
        for metric in ("concurrence", "purity", "bell_fidelity"):
            vals = np.array([r[metric] for r in rs])
            diffs = np.diff(vals)
            bad = np.where(diffs > tol)[0]
            if len(bad) > 0:
                violations.append((T, metric, len(bad), float(diffs[bad].max())))

    fig, ax = plt.subplots(figsize=(7.5, 1.5 + 0.3 * max(len(violations), 1)))
    ax.axis("off")
    lines = [f"Monotonicity check (tolerance={tol:.0e}):"]
    if not violations:
        lines.append("  No violations found -- concurrence/purity/fidelity")
        lines.append("  are non-increasing with delay at every temperature.")
    else:
        for T, metric, n, worst in violations:
            lines.append(
                f"  T={T:.0f}K, {metric}: {n} non-monotonic step(s), "
                f"worst increase={worst:.2e}"
            )
    ax.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=9, family="monospace")
    fig.tight_layout()
    fig.savefig(figs_dir / "monotonicity_check.png", dpi=150)
    plt.close(fig)
    return violations


def run_basic_diagnostics(rows, figs_dir):
    plot_metric_vs_delay(
        rows, "concurrence", "Concurrence", figs_dir,
        "concurrence_vs_delay.png", ylim=(0, 1.05),
    )
    plot_metric_vs_delay(
        rows, "purity", "Purity", figs_dir,
        "purity_vs_delay.png", ylim=(0, 1.05),
    )
    plot_metric_vs_delay(
        rows, "bell_fidelity", "Bell fidelity", figs_dir,
        "bell_fidelity_vs_delay.png", ylim=(0, 1.05),
    )
    plot_metric_vs_delay(
        rows, "mutual_information_bits", "Mutual information (bits)",
        figs_dir, "mutual_information_vs_delay.png",
    )
    plot_metric_vs_delay(
        rows, "l1_coherence", "$l_1$ coherence", figs_dir,
        "l1_coherence_vs_delay.png",
    )
    plot_metric_vs_delay(
        rows, "relative_entropy_coherence_bits",
        "Relative entropy of coherence (bits)", figs_dir,
        "relative_entropy_coherence_vs_delay.png",
    )
    plot_metric_vs_delay(
        rows, "entropy_bits", "von Neumann entropy (bits)", figs_dir,
        "entropy_vs_delay.png",
    )
    plot_degradation_vs_temperature(rows, figs_dir)

    violations = check_monotonicity(rows, figs_dir)
    if violations:
        print(
            f"WARNING: {len(violations)} monotonicity violation(s) detected "
            "-- see figs/monotonicity_check.png"
        )
    else:
        print("Monotonicity check passed: no violations.")


# ---------------------------------------------------------------------
# Section 2: weak-coupling check
# ---------------------------------------------------------------------

def compute_D_M(n_spins, J, delta, temperature, delta_t_array,
                 probe_model, probe_sigma_sites, h_z=0.0, periodic=False):
    """D_M(t) at a single temperature, for an array of dimensionless delays.
    Cheap: a single eigh at the spin-only dimension (2^n_spins), then a
    vectorized sum per delay -- never touches the joint 4*2^n_spins space.
    """
    Hs = build_spin_hamiltonian_xxz(n_spins, J, delta, h_z=h_z, periodic=periodic)
    weights = build_probe_weights(n_spins, probe_model, probe_sigma_sites)
    M = weighted_magnetization_z(n_spins, weights)

    evals, evecs = eigh(Hs)
    M_eig = evecs.conj().T @ M @ evecs

    if temperature <= 0:
        # ground-state (possibly degenerate) equal mixture, matching
        # states.thermal_state_from_hamiltonian's T=0 convention
        mask = np.isclose(evals, evals.min(), atol=1e-10, rtol=0.0)
        p = mask.astype(float) / mask.sum()
    else:
        shifted = (evals - evals.min()) / temperature
        w = np.exp(-shifted)
        p = w / w.sum()

    abs_M2 = np.abs(M_eig) ** 2
    weight_ab = p[:, None] * abs_M2
    E_diff = evals[:, None] - evals[None, :]

    D_M = np.array([
        np.sum(weight_ab * (1.0 - np.cos(E_diff * t))) for t in delta_t_array
    ])
    return D_M


def run_weak_coupling_check(rows, figs_dir):
    groups = _group_by_temperature(rows, sort_key="delta_t")
    theta1, theta2 = cfg.theta1, cfg.theta2
    if not np.isclose(theta1, theta2):
        print(
            f"WARNING: theta1={theta1} != theta2={theta2}; the weak-coupling "
            "formula used here assumes theta1=theta2=theta. Skipping "
            "quantitative overlay, plotting simulated data only."
        )
        theta = None
    else:
        theta = theta1

    temps = sorted(groups.keys())
    cmap = plt.get_cmap("viridis")
    n_temps = max(len(temps) - 1, 1)

    fig_F, ax_F = plt.subplots(figsize=(7.5, 5.5))
    fig_C, ax_C = plt.subplots(figsize=(7.5, 5.5))
    fig_resid, ax_resid = plt.subplots(figsize=(7.5, 5.5))

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
                cfg.N_spins, cfg.J, cfg.delta, T_dimensionless,
                delta_t_dimensionless, cfg.probe_model, cfg.probe_sigma_sites,
                h_z=cfg.h_z, periodic=cfg.periodic,
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
        f"Bell fidelity: simulated vs 2nd-order weak-coupling prediction\n"
        f"({cfg.material}, theta={cfg.theta1:.3g}, N={cfg.N_spins})"
    )
    ax_F.legend(fontsize=6, ncol=2)
    ax_F.grid(alpha=0.3)
    fig_F.tight_layout()
    fig_F.savefig(figs_dir / "bell_fidelity_weak_coupling_check.png", dpi=150)
    plt.close(fig_F)

    ax_C.set_xlabel("delay (fs)")
    ax_C.set_ylabel("Concurrence")
    ax_C.set_title(
        f"Concurrence: simulated vs 2nd-order weak-coupling prediction\n"
        f"({cfg.material}, theta={cfg.theta1:.3g}, N={cfg.N_spins})"
    )
    ax_C.legend(fontsize=6, ncol=2)
    ax_C.grid(alpha=0.3)
    fig_C.tight_layout()
    fig_C.savefig(figs_dir / "concurrence_weak_coupling_check.png", dpi=150)
    plt.close(fig_C)

    ax_resid.set_xlabel("delay (fs)")
    ax_resid.set_ylabel("|F_simulated - F_predicted|")
    ax_resid.set_yscale("log")
    ax_resid.set_title(
        f"Weak-coupling residual (should scale as O(theta^3) = O({cfg.theta1**3:.1e}))\n"
        f"({cfg.material}, theta={cfg.theta1:.3g})"
    )
    ax_resid.legend(fontsize=6, ncol=2)
    ax_resid.grid(alpha=0.3, which="both")
    fig_resid.tight_layout()
    fig_resid.savefig(figs_dir / "weak_coupling_residual.png", dpi=150)
    plt.close(fig_resid)

    if theta is not None:
        theta_cubed = theta ** 3
        print(f"Max |F_sim - F_predicted| across sweep: {max_residual:.3e}")
        print(f"theta^3 = {theta_cubed:.3e} (expected residual order)")
        if max_residual > 50 * theta_cubed:
            print(
                "WARNING: residual is much larger than theta^3 -- either "
                "theta is not small enough for the weak-coupling regime at "
                "these parameters, or there is a discrepancy worth "
                "investigating."
            )
        else:
            print("Residual is consistent with the expected O(theta^3) order.")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    rows, summary_path = _load_summary(path)
    print(f"Loaded {len(rows)} rows from {summary_path}")

    figs_dir = _figs_dir(summary_path)
    print(f"Writing figures to {figs_dir}")

    print("\n--- Section 1: basic diagnostics ---")
    run_basic_diagnostics(rows, figs_dir)

    print("\n--- Section 2: weak-coupling check ---")
    run_weak_coupling_check(rows, figs_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()