"""Reduced states and basic expectation values."""
from __future__ import annotations

import numpy as np


def expectation_value(rho: np.ndarray, operator: np.ndarray) -> complex:
    """Return Tr(rho O) without silently discarding an imaginary part."""
    return np.trace(np.asarray(rho) @ np.asarray(operator))


def partial_trace_spins(rho: np.ndarray, n_spins: int) -> np.ndarray:
    """Trace out all spins, returning the 4x4 two-photon polarization state."""
    dim_s = 2**n_spins
    r = np.asarray(rho, dtype=complex).reshape(4, dim_s, 4, dim_s)
    return np.trace(r, axis1=1, axis2=3)


def partial_trace_photons(rho: np.ndarray, n_spins: int) -> np.ndarray:
    """Trace out the two photon polarization qubits."""
    dim_s = 2**n_spins
    r = np.asarray(rho, dtype=complex).reshape(4, dim_s, 4, dim_s)
    return np.trace(r, axis1=0, axis2=2)


def partial_trace_photon_mode(rho_photons: np.ndarray, trace_out: int) -> np.ndarray:
    """Trace one qubit from a 4x4 two-qubit photon density matrix."""
    if trace_out not in (0, 1):
        raise ValueError("trace_out must be 0 or 1")
    r = np.asarray(rho_photons, dtype=complex).reshape(2, 2, 2, 2)
    if trace_out == 0:
        return np.trace(r, axis1=0, axis2=2)
    return np.trace(r, axis1=1, axis2=3)
