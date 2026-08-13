"""Unitary evolution primitives."""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def unitary_from_generator(generator: np.ndarray, angle: float) -> np.ndarray:
    """Return exp(-i angle * generator)."""
    generator = np.asarray(generator, dtype=complex)
    return expm(-1j * float(angle) * generator)


def unitary_from_hamiltonian(H: np.ndarray, t: float) -> np.ndarray:
    """Return exp(-i H t), with hbar=1."""
    return unitary_from_generator(H, t)


def apply_unitary(rho: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Apply a unitary without artificial renormalization or symmetrization."""
    return U @ rho @ U.conj().T


def apply_unitaries(rho: np.ndarray, unitaries) -> np.ndarray:
    """Apply unitaries in the order supplied."""
    out = np.asarray(rho, dtype=complex)
    for U in unitaries:
        out = apply_unitary(out, np.asarray(U, dtype=complex))
    return out
