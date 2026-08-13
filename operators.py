"""Linear-algebra building blocks for the EP-MOKS polarization-qubit model.

Hilbert-space ordering throughout the repository is
    photon_1 x photon_2 x spin_1 x ... x spin_N,
where each factor is two-dimensional.
"""
from __future__ import annotations

import numpy as np


I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
SP = 0.5 * (SX + 1j * SY)
SM = 0.5 * (SX - 1j * SY)


def kron_all(ops):
    """Kronecker product of a non-empty iterable of matrices."""
    ops = list(ops)
    if not ops:
        raise ValueError("kron_all requires at least one operator")
    out = np.asarray(ops[0], dtype=complex)
    for op in ops[1:]:
        out = np.kron(out, np.asarray(op, dtype=complex))
    return out


def full_dim(n_spins: int) -> int:
    """Dimension of the two-photon-qubit plus N-spin Hilbert space."""
    if n_spins < 1:
        raise ValueError("n_spins must be >= 1")
    return 2 ** (n_spins + 2)


def photon_operator(op: np.ndarray, photon: int, n_spins: int) -> np.ndarray:
    """Embed a 2x2 operator on photon 0 or 1 in the full Hilbert space."""
    if photon not in (0, 1):
        raise ValueError("photon must be 0 or 1")
    factors = [I2, I2] + [I2] * n_spins
    factors[photon] = np.asarray(op, dtype=complex)
    return kron_all(factors)


def spin_operator(op: np.ndarray, site: int, n_spins: int) -> np.ndarray:
    """Embed a 2x2 operator on one spin site in the full Hilbert space."""
    if not 0 <= site < n_spins:
        raise ValueError(f"site={site} outside [0, {n_spins})")
    factors = [I2, I2] + [I2] * n_spins
    factors[2 + site] = np.asarray(op, dtype=complex)
    return kron_all(factors)


def spin_only_operator(op: np.ndarray, site: int, n_spins: int) -> np.ndarray:
    """Embed a 2x2 operator on one site in the spin-only Hilbert space."""
    if not 0 <= site < n_spins:
        raise ValueError(f"site={site} outside [0, {n_spins})")
    factors = [I2] * n_spins
    factors[site] = np.asarray(op, dtype=complex)
    return kron_all(factors)


def embed_spin_only(op_spin: np.ndarray) -> np.ndarray:
    """Embed a spin-only operator after the two photon-qubit factors."""
    op_spin = np.asarray(op_spin, dtype=complex)
    return np.kron(np.eye(4, dtype=complex), op_spin)
