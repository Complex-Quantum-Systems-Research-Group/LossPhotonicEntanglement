"""State diagnostics used by the corrected EP-MOKS workflow."""
from __future__ import annotations

import numpy as np

from observables import partial_trace_photon_mode
from operators import SY


def _normalized_hermitian(rho: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, dtype=complex)
    return 0.5 * (rho + rho.conj().T)


def von_neumann_entropy(rho: np.ndarray, base: float = 2.0, tol: float = 1e-12) -> float:
    """Von Neumann entropy.  For the photon pair this quantifies mixedness, not entanglement."""
    vals = np.linalg.eigvalsh(_normalized_hermitian(rho)).real
    vals = np.clip(vals, 0.0, None)
    s = vals.sum()
    if s <= tol:
        raise ValueError("density matrix has vanishing trace")
    vals /= s
    vals = vals[vals > tol]
    logs = np.log(vals) / np.log(base)
    return float(-np.sum(vals * logs))


def purity(rho: np.ndarray) -> float:
    """Tr(rho^2)."""
    rho = np.asarray(rho, dtype=complex)
    return float(np.trace(rho @ rho).real)


def mutual_information(rho_photons: np.ndarray) -> float:
    """Total (classical + quantum) correlation between photon polarization qubits."""
    rho1 = partial_trace_photon_mode(rho_photons, trace_out=1)
    rho2 = partial_trace_photon_mode(rho_photons, trace_out=0)
    return von_neumann_entropy(rho1) + von_neumann_entropy(rho2) - von_neumann_entropy(rho_photons)


def l1_coherence(rho: np.ndarray) -> float:
    """l1-norm coherence in the computational H/V basis."""
    rho = np.asarray(rho, dtype=complex)
    return float(np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho))))


def relative_entropy_coherence(rho: np.ndarray) -> float:
    """Relative entropy of coherence in the computational H/V basis."""
    rho = np.asarray(rho, dtype=complex)
    diag = np.diag(np.diag(rho))
    return von_neumann_entropy(diag) - von_neumann_entropy(rho)


def concurrence(rho: np.ndarray, tol: float = 1e-12) -> float:
    """Wootters concurrence for a two-qubit density matrix."""
    rho = _normalized_hermitian(rho)
    if rho.shape != (4, 4):
        raise ValueError("concurrence requires a 4x4 two-qubit density matrix")
    yy = np.kron(SY, SY)
    rho_tilde = yy @ rho.conj() @ yy
    vals = np.linalg.eigvals(rho @ rho_tilde)
    vals = np.sort(np.sqrt(np.clip(vals.real, 0.0, None)))[::-1]
    c = vals[0] - vals[1] - vals[2] - vals[3]
    if abs(c) < tol:
        c = 0.0
    return float(np.clip(c, 0.0, 1.0))


def bell_fidelity(rho: np.ndarray, bell_rho: np.ndarray) -> float:
    """Fidelity with a pure target Bell state represented as a rank-1 density matrix."""
    return float(np.trace(np.asarray(rho) @ np.asarray(bell_rho)).real)
