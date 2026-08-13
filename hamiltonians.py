"""Hamiltonians and interaction generators for the corrected EP-MOKS model.

The primary model treats each probe photon as a polarization qubit and the MOKE
interaction as a short, magnetization-conditioned polarization rotation.  This
is a controlled toy model for Kerr rotation, not a microscopic electronic MOKE
Hamiltonian.
"""
from __future__ import annotations

import numpy as np

from operators import I2, SX, SY, SZ, SP, SM, kron_all, spin_only_operator


def build_spin_hamiltonian_xxz(
    n_spins: int,
    J: float,
    delta: float,
    h_z: float = 0.0,
    periodic: bool = False,
) -> np.ndarray:
    r"""Return the spin-only XXZ Hamiltonian.

    We use the conventional spin-1/2 operators S = sigma/2:

        H = J sum_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1}
                     + Delta Sz_i Sz_{i+1}) - h_z sum_i Sz_i.

    Therefore the Pauli-matrix representation carries an overall factor 1/4 on
    the exchange terms and 1/2 on the Zeeman term.  Units use hbar = k_B = 1.
    """
    if n_spins < 2:
        raise ValueError("n_spins must be >= 2 for an XXZ chain")

    dim = 2**n_spins
    H = np.zeros((dim, dim), dtype=complex)

    bonds = [(i, i + 1) for i in range(n_spins - 1)]
    if periodic and n_spins > 2:
        bonds.append((n_spins - 1, 0))

    for i, j in bonds:
        sx_i = spin_only_operator(SX, i, n_spins)
        sx_j = spin_only_operator(SX, j, n_spins)
        sy_i = spin_only_operator(SY, i, n_spins)
        sy_j = spin_only_operator(SY, j, n_spins)
        sz_i = spin_only_operator(SZ, i, n_spins)
        sz_j = spin_only_operator(SZ, j, n_spins)
        H += (J / 4.0) * (sx_i @ sx_j + sy_i @ sy_j + delta * (sz_i @ sz_j))

    if h_z != 0.0:
        for i in range(n_spins):
            H -= (h_z / 2.0) * spin_only_operator(SZ, i, n_spins)

    return 0.5 * (H + H.conj().T)


def probe_weights_gaussian(n_spins: int, center: float | None = None, sigma: float = 1.0) -> np.ndarray:
    """Normalized nonuniform probe weights for a localized optical spot."""
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if center is None:
        center = 0.5 * (n_spins - 1)
    x = np.arange(n_spins, dtype=float)
    w = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    return w / np.sum(w)


def weighted_magnetization_z(n_spins: int, weights=None) -> np.ndarray:
    r"""Return a dimensionless weighted z magnetization in the spin subspace.

    M_z = sum_i w_i sigma_z^(i), with sum_i |w_i| = 1 by normalization.
    Its operator norm is <= 1 for nonnegative normalized weights.

    Equal weights produce the conserved collective magnetization of the XXZ
    chain.  Nonuniform weights model a finite/local optical spot and generally
    do not commute with the exchange Hamiltonian, allowing delay-time dynamics.
    """
    if weights is None:
        w = np.ones(n_spins, dtype=float) / n_spins
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n_spins,):
            raise ValueError(f"weights must have shape ({n_spins},)")
        norm = np.sum(np.abs(w))
        if norm <= 0:
            raise ValueError("weights must not all vanish")
        w = w / norm

    M = np.zeros((2**n_spins, 2**n_spins), dtype=complex)
    for i, wi in enumerate(w):
        M += wi * spin_only_operator(SZ, i, n_spins)
    return 0.5 * (M + M.conj().T)


def collective_magnetization_z(n_spins: int) -> np.ndarray:
    """Convenience wrapper for equally weighted collective z magnetization."""
    return weighted_magnetization_z(n_spins, np.ones(n_spins, dtype=float))


def exchange_generator_spin_only(n_spins: int, weights=None) -> tuple[np.ndarray, np.ndarray]:
    """Weighted collective spin raising/lowering operators for a benchmark model."""
    if weights is None:
        w = np.ones(n_spins, dtype=float) / n_spins
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n_spins,):
            raise ValueError(f"weights must have shape ({n_spins},)")
        norm = np.sum(np.abs(w))
        if norm <= 0:
            raise ValueError("weights must not all vanish")
        w = w / norm

    sp = np.zeros((2**n_spins, 2**n_spins), dtype=complex)
    sm = np.zeros_like(sp)
    for i, wi in enumerate(w):
        sp += wi * spin_only_operator(SP, i, n_spins)
        sm += wi * spin_only_operator(SM, i, n_spins)
    return sp, sm


def kerr_interaction_generator(n_spins: int, photon: int, magnetization: np.ndarray) -> np.ndarray:
    r"""Generator G_k = sigma_y^(photon k) tensor M_z for Kerr rotations.

    U_k(theta) = exp(-i theta G_k).

    In the H/V Jones basis, exp(-i theta sigma_y) is a real polarization
    rotation.  The material operator M_z makes the rotation conditional on the
    sampled magnetization.
    """
    if photon not in (0, 1):
        raise ValueError("photon must be 0 or 1")
    magnetization = np.asarray(magnetization, dtype=complex)
    if magnetization.shape != (2**n_spins, 2**n_spins):
        raise ValueError("magnetization has incompatible shape")
    p_factors = [SY if photon == 0 else I2, SY if photon == 1 else I2]
    return kron_all(p_factors + [magnetization])


def exchange_interaction_generator(n_spins: int, photon: int, weights=None) -> np.ndarray:
    r"""Generic excitation-exchange benchmark for a photon polarization qubit.

    G = sigma_+^(p) tensor S_- + sigma_-^(p) tensor S_+.

    This is retained only as a generic coherent exchange benchmark.  It is not
    identified with the magneto-optical Kerr interaction.
    """
    if photon not in (0, 1):
        raise ValueError("photon must be 0 or 1")
    sp_s, sm_s = exchange_generator_spin_only(n_spins, weights)
    pplus = SP
    pminus = SM
    p0_plus = pplus if photon == 0 else I2
    p1_plus = pplus if photon == 1 else I2
    p0_minus = pminus if photon == 0 else I2
    p1_minus = pminus if photon == 1 else I2
    return kron_all([p0_plus, p1_plus, sm_s]) + kron_all([p0_minus, p1_minus, sp_s])
