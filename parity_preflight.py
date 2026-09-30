"""Preflight controls for the Bell-diagonality/parity claim.

Computes the exact spin-only characteristic function

    C21 = Tr[rho_S exp(-i theta1 M0) exp(2 i theta2 Mt)
              exp(-i theta1 M0)]

and, optionally, the corresponding photon-state diagnostics from the full
protocol.  This is intentionally a small-point diagnostic, not a sweep.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.linalg import eigh, expm

from hamiltonians import build_spin_hamiltonian_xxz, weighted_magnetization_z
from measures import l1_coherence, relative_entropy_coherence
from new_protocol import build_probe_weights, full_pipeline_unitary
from observables import partial_trace_spins
from states import thermal_state_from_hamiltonian


PHI_PLUS = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2.0)
PSI_MINUS = np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / np.sqrt(2.0)


def analytic_consequences(C: complex) -> tuple[float, float, float]:
    """Return ``(p, |<Phi+|rho|Psi->|, C_l1)`` from ImCconsequences."""
    C = complex(C)
    return (1.0 - C.real) / 2.0, abs(C.imag) / 2.0, 1.0 + 2.0 * abs(C.imag)


def spin_characteristic_C(
    n_spins: int,
    J: float,
    delta: float,
    h_z: float,
    temperature: float,
    delay: float,
    theta1: float,
    theta2: float,
    probe_model: str = "local_gaussian",
    probe_sigma_sites: float = 1.0,
    periodic: bool = False,
) -> complex:
    """Return the exact spin-only C21 characteristic function."""
    H = build_spin_hamiltonian_xxz(
        n_spins, J, delta, h_z=h_z, periodic=periodic
    )
    rho_s = thermal_state_from_hamiltonian(H, temperature)
    weights = build_probe_weights(n_spins, probe_model, probe_sigma_sites)
    M0 = weighted_magnetization_z(n_spins, weights)

    evals, evecs = eigh(H)
    phases = np.exp(-1j * evals * delay)
    U = (evecs * phases) @ evecs.conj().T
    Mt = U.conj().T @ M0 @ U

    expression = (
        expm(-1j * theta1 * M0)
        @ expm(2j * theta2 * Mt)
        @ expm(-1j * theta1 * M0)
    )
    return complex(np.trace(rho_s @ expression))


def photon_diagnostics(**kwargs) -> dict[str, float]:
    """Return the three diagnostics requested for the parity preflight."""
    n_spins = int(kwargs["n_spins"])
    rho_full = full_pipeline_unitary(**kwargs)
    rho_p = partial_trace_spins(rho_full, n_spins)
    cross = np.vdot(PHI_PLUS, rho_p @ PSI_MINUS)
    return {
        "bell_fidelity": float(np.vdot(PHI_PLUS, rho_p @ PHI_PLUS).real),
        "C_l1": l1_coherence(rho_p),
        "C_rel": relative_entropy_coherence(rho_p),
        "abs_phi_plus_rho_psi_minus": float(abs(cross)),
        "numerical_rank": int(np.count_nonzero(np.linalg.eigvalsh(rho_p) > 1e-12)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-spins", type=int, default=8)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.05, 0.76])
    parser.add_argument("--fields", type=float, nargs="+", default=[0.0, 0.1, 0.5])
    parser.add_argument("--delays", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0, 4.0])
    parser.add_argument("--theta", type=float, default=0.4)
    parser.add_argument("--sigma", type=float, default=1.0)
    args = parser.parse_args()

    rows = []
    for h_z in args.fields:
        for temperature in args.temperatures:
            for delay in args.delays:
                common = dict(
                    n_spins=args.n_spins, J=1.0, delta=1.0,
                    h_z=h_z, temperature=temperature, delta_t=delay,
                    theta1=args.theta, theta2=args.theta,
                    probe_model="local_gaussian", probe_sigma_sites=args.sigma,
                    periodic=False, interaction_type="kerr", bell_state="phi_plus",
                )
                C = spin_characteristic_C(
                    delay=common["delta_t"],
                    **{k: v for k, v in common.items() if k not in {
                        "delta_t", "interaction_type", "bell_state"
                    }},
                )
                row = {"h_z": h_z, "temperature": temperature, "delay": delay,
                       "Re_C": C.real, "Im_C": C.imag}
                diagnostics = photon_diagnostics(**common)
                p_a, coh_a, cl1_a = analytic_consequences(C)
                if not np.isclose(diagnostics["bell_fidelity"], 1.0 - p_a, atol=1e-12, rtol=0.0):
                    raise AssertionError("Bell fidelity violates analytic C identity")
                if not np.isclose(
                    diagnostics["abs_phi_plus_rho_psi_minus"], coh_a,
                    atol=1e-12, rtol=0.0,
                ):
                    raise AssertionError("Bell cross coherence violates analytic C identity")
                if not np.isclose(diagnostics["C_l1"], cl1_a, atol=1e-12, rtol=0.0):
                    raise AssertionError("l1 coherence violates analytic C identity")
                if diagnostics["numerical_rank"] > 2:
                    raise AssertionError("photon state violates unconditional rank-two support")
                row.update({
                    **diagnostics,
                    "analytic_p": p_a,
                    "analytic_abs_phi_plus_rho_psi_minus": coh_a,
                    "analytic_C_l1": cl1_a,
                })
                rows.append(row)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
