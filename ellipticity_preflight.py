"""Exact elliptical Kerr controls, independent of campaign parameters."""
from __future__ import annotations

import json
import numpy as np
from scipy.linalg import expm

from hamiltonians import build_spin_hamiltonian_xxz, weighted_magnetization_z
from new_evolution import apply_kron_sum_sequence, delay_terms, kerr_rotation_terms
from operators import I2, SY, SZ
from states import bell_polarization_state, thermal_state_from_hamiltonian
from observables import partial_trace_spins
from validation import assert_density_matrix, assert_rank_two_support, assert_unitality


def zero_delay_bell_fidelity(theta1, eta1, theta2, eta2, magnetization, rho_spin):
    """Exact Phi+ fidelity at zero delay for arbitrary real coupling angles.

    Spectral averaging of |Tr[U1(m) U2(m)^T]/2|^2. The equal-angle
    case is the manuscript's O(m)^2 formula. sinc handles zero coupling.
    No assumption that rho_spin commutes with M is needed.
    """
    m, vectors = np.linalg.eigh(magnetization)
    q = np.diag(vectors.conj().T @ rho_spin @ vectors).real
    a, b = np.hypot(theta1, eta1), np.hypot(theta2, eta2)
    s1 = m * np.sinc(m * a / np.pi)
    s2 = m * np.sinc(m * b / np.pi)
    overlap = np.cos(m * a) * np.cos(m * b) + (theta1 * theta2 - eta1 * eta2) * s1 * s2
    return float(q @ overlap**2)


def validate_ellipticity():
    """Reproduce the manuscript's N=3, T=0.7, w=(0.5,0.3,0.2) checks."""
    n = 3
    H = build_spin_hamiltonian_xxz(n, 1.0, 1.0)
    M = weighted_magnetization_z(n, [0.5, 0.3, 0.2])
    rho_s = thermal_state_from_hamiltonian(H, 0.7)
    rho0 = bell_polarization_state("phi_plus")

    def run(t1, e1, t2, e2, dt, photon=rho0):
        terms = (kerr_rotation_terms(t1, 0, M, eta=e1), delay_terms(H, dt),
                 kerr_rotation_terms(t2, 1, M, eta=e2))
        full = apply_kron_sum_sequence(np.kron(photon, rho_s), terms)
        # Independent dense exponentials: bypass projector and contraction code.
        U1 = expm(-1j * np.kron(np.kron(t1 * SY + e1 * SZ, I2), M))
        U2 = expm(-1j * np.kron(np.kron(I2, t2 * SY + e2 * SZ), M))
        U = U2 @ np.kron(np.eye(4), expm(-1j * H * dt)) @ U1
        reference = U @ np.kron(photon, rho_s) @ U.conj().T
        if np.linalg.norm(full - reference) > 1e-10:
            raise AssertionError("elliptical fast/dense propagation mismatch")
        return partial_trace_spins(full, n)

    identity = run(.35, .22, .28, .11, 1.7, photon=np.eye(4))
    unitality = assert_unitality(identity)
    rotation = run(.4, 0., .4, 0., 1.7)
    elliptical = run(.4, .25, .4, .25, 1.7)
    assert_rank_two_support(rotation)
    assert_density_matrix(elliptical)
    ranks = [int(np.count_nonzero(np.linalg.eigvalsh(r) > 1e-9))
             for r in (rotation, elliptical)]
    if ranks != [2, 4]:
        raise AssertionError(f"expected ranks 2 and 4 at reference point, got {ranks}")
    ratios = [float(np.linalg.norm(run(e, .6*e, e, .6*e, 1.7) - rho0) / e**2)
              for e in (1e-2, 1e-3, 1e-4)]
    if not np.allclose(ratios, 0.6074, atol=1e-4, rtol=0):
        raise AssertionError(f"quadratic response check failed: {ratios}")
    numerical = float(np.trace(rho0 @ run(.4, .25, .4, .25, 0.)).real)
    predicted = zero_delay_bell_fidelity(.4, .25, .4, .25, M, rho_s)
    if abs(numerical - predicted) > 1e-10:
        raise AssertionError("elliptical zero-delay fidelity mismatch")
    return dict(unitality_residual=unitality, ranks=ranks, quadratic_ratios=ratios,
                zero_delay_fidelity=numerical, zero_delay_prediction=predicted,
                zero_delay_error=abs(numerical-predicted), tolerance=1e-10)


if __name__ == "__main__":
    print(json.dumps(validate_ellipticity(), indent=2))
