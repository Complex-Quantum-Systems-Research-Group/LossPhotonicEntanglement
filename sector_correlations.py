"""Exact U(1)-sector evaluation of the spin-only magnetization distance."""
from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
from scipy.linalg import eigh

from new_protocol import build_probe_weights


@dataclass
class SectorSpectrum:
    k: int
    multiplicity: int
    energies: np.ndarray
    matrix_element_sq: np.ndarray


def _sector_operators(n, k, J, delta, h_z, periodic, weights):
    basis = [state for state in range(1 << n) if state.bit_count() == k]
    index = {state: i for i, state in enumerate(basis)}
    dim = len(basis)
    H = np.zeros((dim, dim), dtype=float)
    mdiag = np.empty(dim, dtype=float)
    bonds = [(i, i + 1) for i in range(n - 1)]
    if periodic and n > 2:
        bonds.append((n - 1, 0))

    def z(state, site):
        return 1.0 if not (state >> (n - 1 - site)) & 1 else -1.0

    for row, state in enumerate(basis):
        zs = np.fromiter((z(state, site) for site in range(n)), float, count=n)
        H[row, row] = sum(J * delta * zs[i] * zs[j] / 4.0 for i, j in bonds)
        H[row, row] -= h_z * float(zs.sum()) / 2.0
        mdiag[row] = float(weights @ zs)
        for i, j in bonds:
            if zs[i] != zs[j]:
                flipped = state ^ (1 << (n - 1 - i)) ^ (1 << (n - 1 - j))
                H[row, index[flipped]] += J / 2.0
    return H, mdiag


def diagonalize_magnetization_sectors(
    n_spins, J, delta, probe_model="local_gaussian", probe_sigma_sites=1.0,
    h_z=0.0, periodic=False, use_spin_flip=True,
):
    """Diagonalize conserved Hamming-weight sectors and cache |M_ab|^2."""
    if use_spin_flip and h_z != 0.0:
        raise ValueError("spin-flip sector pairing requires h_z=0")
    weights = build_probe_weights(n_spins, probe_model, probe_sigma_sites)
    stop = n_spins // 2 if use_spin_flip else n_spins
    sectors = []
    for k in range(stop + 1):
        H, mdiag = _sector_operators(
            n_spins, k, J, delta, h_z, periodic, weights
        )
        energies, vectors = eigh(H, overwrite_a=True, check_finite=False)
        m_eigen = vectors.T @ (mdiag[:, None] * vectors)
        multiplicity = 2 if use_spin_flip and k != n_spins - k else 1
        sectors.append(SectorSpectrum(k, multiplicity, energies, m_eigen * m_eigen))
    assert sum(s.multiplicity * len(s.energies) for s in sectors) == 2**n_spins
    return sectors, weights


def sector_D_M(
    sectors, temperature, delays, relative_cutoff=1e-14,
    max_outer_elements=12_000_000,
):
    """Evaluate exact D_M with one global partition function and delay chunks."""
    delays = np.asarray(delays, dtype=float)
    emin = min(float(s.energies[0]) for s in sectors)
    if temperature < 0:
        raise ValueError("temperature must be nonnegative")
    if temperature == 0:
        degeneracy = sum(
            s.multiplicity * int(np.count_nonzero(np.isclose(s.energies, emin, atol=1e-12, rtol=0)))
            for s in sectors
        )
        populations = [np.isclose(s.energies, emin, atol=1e-12, rtol=0).astype(float) / degeneracy for s in sectors]
        Z = float(degeneracy)
    else:
        raw = [np.exp(-(s.energies - emin) / temperature) for s in sectors]
        Z = float(sum(s.multiplicity * w.sum() for s, w in zip(sectors, raw)))
        populations = [w / Z for w in raw]

    result = np.zeros_like(delays)
    s0 = 0.0
    kept = total = 0
    for sector, pop in zip(sectors, populations):
        A = pop[:, None] * sector.matrix_element_sq
        total += A.size
        threshold = relative_cutoff * float(A.max(initial=0.0))
        mask = A >= threshold if threshold > 0 else A > 0
        avec = A[mask].ravel()
        omega = (sector.energies[:, None] - sector.energies[None, :])[mask].ravel()
        kept += len(avec)
        mult = sector.multiplicity
        s0 += mult * float(avec.sum())
        chunk = max(1, max_outer_elements // max(1, len(avec)))
        for start in range(0, len(delays), chunk):
            sl = slice(start, min(start + chunk, len(delays)))
            result[sl] += mult * (np.cos(np.outer(delays[sl], omega)) @ avec)
    dm = s0 - result
    dm[np.isclose(delays, 0.0, atol=0.0, rtol=0.0)] = 0.0
    return dm, {"Emin": emin, "Z_shifted": Z, "S0": s0, "kept_pairs": kept, "total_pairs": total}


def infinite_temperature_M2(weights):
    return float(np.sum(np.asarray(weights, dtype=float) ** 2))
