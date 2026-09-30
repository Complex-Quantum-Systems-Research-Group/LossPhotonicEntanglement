"""Checks the branch-complementation collapse of the local-thermometry QFI
optimum and its origin in the h_z=0 magnetization-inversion symmetry R.

See local manuscript/notes.md for the derivation and the full-grid results
against the saved N=10 campaigns. This script only re-derives the mechanism
on small systems (fast exact diagonalization) and reports residuals; it does
not repeat the full-grid check against reports/local_thermometry, which
requires the saved N=10 kernels.npz caches and is done inline where those
caches are read.

Branch order is (s,t) in {0,1}^2 with index = 2*s+t (s: photon-1 sign index,
t: photon-2 sign index). Complementation is Pi: (s,t) -> (1-s,1-t), i.e. the
index permutation [3, 2, 1, 0].
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from new_protocol import build_probe_weights
from sector_correlations import _sector_operators
from thermometry import ThermometryChannel, optimize_inputs, quantum_fisher_information

PERM = [3, 2, 1, 0]


def qfi_for_w(K, dK, w):
    """QFI of the one-parameter family: equal weight (1-w)/2 on branches
    {00,11}, w/2 on branches {01,10}, real amplitudes (phase-irrelevant)."""
    a = np.array([np.sqrt((1 - w) / 2), np.sqrt(w / 2), np.sqrt(w / 2), np.sqrt((1 - w) / 2)])
    state = np.outer(a, a)
    return quantum_fisher_information(K * state, dK * state, cutoff=1e-13)


def best_w(K, dK, n=2001):
    ws = np.linspace(0, 1, n)
    vals = [qfi_for_w(K, dK, w) for w in ws]
    i = int(np.argmax(vals))
    return ws[i], vals[i]


def sector_mirror_check(n, h_z):
    """Sector k and sector n-k share a Hamiltonian spectrum iff h_z=0 (the
    global spin-inversion symmetry R). h_z splits each pair by h_z*(n-2k),
    the Zeeman shift between the two sectors' total magnetizations."""
    print(f"  sector mirror-energy check (N={n}, h_z={h_z}):")
    for k in range(n // 2 + 1):
        h1, _ = _sector_operators(n, k, 1.0, 1.0, h_z, False, np.ones(n) / n)
        h2, _ = _sector_operators(n, n - k, 1.0, 1.0, h_z, False, np.ones(n) / n)
        e1 = np.sort(eigh(h1, eigvals_only=True))
        e2 = np.sort(eigh(h2, eigvals_only=True))
        dev = np.abs(e1 - e2).max()
        predicted = abs(h_z * (n - 2 * k))
        print(f"    k={k} vs N-k={n - k}: max energy mismatch={dev:.3e}  "
              f"(predicted Zeeman split h_z*(N-2k)={predicted:.3e})")


def kernel_symmetry_and_collapse(n, h_z, temperatures, delays, eta=0.0, grid_size=41):
    weights = build_probe_weights(n, "local_gaussian", 1.0)
    channel = ThermometryChannel(n, weights, theta1=0.4, theta2=0.4, eta1=eta, eta2=eta,
                                  delta=1.0, h_over_J=h_z, periodic=False)
    print(f"  kernel Pi-invariance and 1-parameter-family gap (N={n}, eta={eta}, h_z={h_z}):")
    max_dev_K, max_dev_dK, max_rel_gap = 0.0, 0.0, 0.0
    worst = None
    for T in temperatures:
        for delay in delays:
            K, dK = channel.kernel(T, delay)
            dev_K = np.abs(K[np.ix_(PERM, PERM)] - K).max()
            dev_dK = np.abs(dK[np.ix_(PERM, PERM)] - dK).max()
            max_dev_K, max_dev_dK = max(max_dev_K, dev_K), max(max_dev_dK, dev_dK)
            _, F_w = best_w(K, dK)
            row = optimize_inputs(K, dK, channel.q_bell, grid_size=grid_size)
            F_all = row["F_all_inputs_found"]
            rel_gap = (F_all - F_w) / max(F_all, 1e-14)
            if rel_gap > max_rel_gap:
                max_rel_gap, worst = rel_gap, (T, delay, row["q_all_opt"])
    print(f"    max|K[Pi,Pi]-K| = {max_dev_K:.3e}   max|dK[Pi,Pi]-dK| = {max_dev_dK:.3e}")
    print(f"    max relative gap of full-simplex optimum over 1-parameter family: {max_rel_gap:.3e}"
          + (f"  (at T={worst[0]}, delay={worst[1]}, q_all_opt={np.round(worst[2], 4)})" if worst else ""))
    return max_dev_K, max_rel_gap


def escape_at_point(n, eta, T, delay, hz_values, grid_size=61):
    """Track escape vs h_z at one fixed (T,delay): distinguishes a smooth,
    continuously growing escape from a level-crossing (threshold) escape,
    both of which are legitimate symmetry-breaking phenomenology."""
    weights = build_probe_weights(n, "local_gaussian", 1.0)
    print(f"  escape vs h_z at N={n}, eta={eta}, T={T}, delay={delay}:")
    for h_z in hz_values:
        channel = ThermometryChannel(n, weights, theta1=0.4, theta2=0.4, eta1=eta, eta2=eta,
                                      delta=1.0, h_over_J=h_z, periodic=False)
        K, dK = channel.kernel(T, delay)
        _, F_w = best_w(K, dK)
        row = optimize_inputs(K, dK, channel.q_bell, grid_size=grid_size)
        F_all = row["F_all_inputs_found"]
        rel_gap = (F_all - F_w) / max(F_all, 1e-14)
        print(f"    h_z={h_z:6.3f}  rel_escape={rel_gap:11.4e}  q_all_opt={np.round(row['q_all_opt'], 4)}")


if __name__ == "__main__":
    N = 6
    temperatures = [0.19, 0.38, 0.76, 1.27]
    delays = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]

    print("=== Sector mirror-energy check: R = global spin inversion, k <-> N-k ===")
    sector_mirror_check(N, h_z=0.0)
    sector_mirror_check(N, h_z=0.5)

    print("\n=== h_z=0: exact Pi-invariance and exact 1-parameter collapse (control) ===")
    kernel_symmetry_and_collapse(N, h_z=0.0, temperatures=temperatures, delays=delays)

    print("\n=== h_z=0.5: broken R, broken Pi-invariance, collapse fails where it matters ===")
    kernel_symmetry_and_collapse(N, h_z=0.5, temperatures=temperatures, delays=delays)

    print("\n=== h_z scan: is escape continuous in h_z, or erratic? ===")
    print("  -- smooth, monotonic escape (soft symmetry breaking) --")
    escape_at_point(N, 0.0, 0.38, 5.0, [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    escape_at_point(N, 0.0, 0.76, 6.0, [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    print("  -- threshold (level-crossing) escape, bisected --")
    escape_at_point(N, 0.0, 0.19, 0.5, [0.40, 0.42, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50])

    print("\n=== Production-scale (N=10) spot check, with and without ellipticity ===")
    prod_temperatures = [0.12672549, 0.25345098, 0.5069019600000001, 0.76035293]
    prod_delays = [3.0, 8.6, 9.2, 11.5]
    for eta in (0.0, 0.25):
        for h_z in (0.0, 0.1, 0.3, 0.5):
            kernel_symmetry_and_collapse(10, h_z=h_z, eta=eta,
                                          temperatures=prod_temperatures, delays=prod_delays)
