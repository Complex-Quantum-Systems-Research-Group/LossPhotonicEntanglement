"""Initial states for EP-MOKS simulations."""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from hamiltonians import build_spin_hamiltonian_xxz


def bell_polarization_state(label: str = "phi_plus") -> np.ndarray:
    """Return a two-polarization-qubit Bell-state density matrix.

    Computational basis mapping: |0> = |H>, |1> = |V>.
    """
    bell = {
        "phi_plus": np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
        "phi_minus": np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),
        "psi_plus": np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),
        "psi_minus": np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2),
    }
    if label not in bell:
        raise ValueError(f"unknown Bell state {label!r}")
    psi = bell[label]
    return np.outer(psi, psi.conj())


def product_polarization_state(first: int = 0, second: int = 0) -> np.ndarray:
    """Return |first, second><first, second| for first,second in {0,1}."""
    if first not in (0, 1) or second not in (0, 1):
        raise ValueError("first and second must be 0 or 1")
    psi = np.zeros(4, dtype=complex)
    psi[2 * first + second] = 1.0
    return np.outer(psi, psi.conj())


def thermal_state_from_hamiltonian(H: np.ndarray, temperature: float, atol: float = 1e-12) -> np.ndarray:
    """Stable Gibbs state exp(-H/T)/Z in units k_B=1.

    At T=0 the canonical limit is taken as the equal mixture on the degenerate
    ground-state subspace.
    """
    H = np.asarray(H, dtype=complex)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be square")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")

    evals, evecs = eigh(0.5 * (H + H.conj().T))
    if temperature == 0:
        mask = np.isclose(evals, evals[0], atol=atol, rtol=0.0)
        V = evecs[:, mask]
        rho = V @ V.conj().T / np.sum(mask)
    else:
        shifted = evals - evals.min()
        weights = np.exp(-shifted / temperature)
        weights /= weights.sum()
        rho = (evecs * weights) @ evecs.conj().T

    rho = 0.5 * (rho + rho.conj().T)
    rho /= np.trace(rho).real
    return rho


def thermal_spin_density_matrix(
    n_spins: int,
    J: float,
    delta: float,
    temperature: float = 1.0,
    h_z: float = 0.0,
    periodic: bool = False,
) -> np.ndarray:
    """Thermal density matrix of the XXZ spin chain."""
    H = build_spin_hamiltonian_xxz(n_spins, J, delta, h_z=h_z, periodic=periodic)
    return thermal_state_from_hamiltonian(H, temperature)
