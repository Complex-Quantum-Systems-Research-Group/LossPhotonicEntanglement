"""Physics/numerics validation helpers."""
from __future__ import annotations

import numpy as np


def assert_unitality(image_identity, atol=1e-10):
    """Kerr implementation control for all rotation and ellipticity angles."""
    residual = float(np.linalg.norm(image_identity - np.eye(4)))
    if residual > atol:
        raise AssertionError(f"unitality residual {residual} exceeds {atol}")
    return residual


def assert_rank_two_support(rho_photons, atol=1e-10):
    """Phi+/Psi- support control; call only for rotation-only Phi+ input."""
    phi_minus = np.array([1, 0, 0, -1]) / np.sqrt(2)
    psi_plus = np.array([0, 1, 1, 0]) / np.sqrt(2)
    # Check full forbidden rows/columns, including coherences.
    for vector in (phi_minus, psi_plus):
        if np.linalg.norm(rho_photons @ vector) > atol:
            raise AssertionError("rotation-only Bell support violated")


def density_matrix_diagnostics(rho: np.ndarray) -> dict:
    rho = np.asarray(rho, dtype=complex)
    herm_err = float(np.linalg.norm(rho - rho.conj().T))
    trace = np.trace(rho)
    evals = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T)).real
    return {
        "hermiticity_error": herm_err,
        "trace_real": float(trace.real),
        "trace_imag": float(trace.imag),
        "min_eigenvalue": float(evals.min()),
        "max_eigenvalue": float(evals.max()),
    }


def assert_density_matrix(rho: np.ndarray, atol: float = 1e-10) -> None:
    d = density_matrix_diagnostics(rho)
    if d["hermiticity_error"] > atol:
        raise AssertionError(f"non-Hermitian density matrix: {d}")
    if abs(d["trace_real"] - 1.0) > atol or abs(d["trace_imag"]) > atol:
        raise AssertionError(f"trace violation: {d}")
    if d["min_eigenvalue"] < -atol:
        raise AssertionError(f"positivity violation: {d}")


def assert_unitary(U: np.ndarray, atol: float = 1e-10) -> None:
    U = np.asarray(U, dtype=complex)
    I = np.eye(U.shape[0], dtype=complex)
    err = np.linalg.norm(U.conj().T @ U - I)
    if err > atol:
        raise AssertionError(f"unitarity error {err}")
