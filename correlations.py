"""Spin-only magnetization correlators used by analysis and validation."""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from hamiltonians import build_spin_hamiltonian_xxz, weighted_magnetization_z
from new_protocol import build_probe_weights


def spectral_magnetization_correlators(
    n_spins, J, delta, temperature, delta_t_array,
    probe_model="local_gaussian", probe_sigma_sites=1.0,
    h_z=0.0, periodic=False,
):
    """Return ``<M(t)M(0)>`` and ``<M(0)M(t)>`` from the energy basis."""
    Hs = build_spin_hamiltonian_xxz(n_spins, J, delta, h_z=h_z, periodic=periodic)
    weights = build_probe_weights(n_spins, probe_model, probe_sigma_sites)
    M = weighted_magnetization_z(n_spins, weights)
    evals, evecs = eigh(Hs)
    M_eig = evecs.conj().T @ M @ evecs

    if temperature <= 0:
        mask = np.isclose(evals, evals.min(), atol=1e-10, rtol=0.0)
        populations = mask.astype(float) / mask.sum()
    else:
        boltzmann = np.exp(-(evals - evals.min()) / temperature)
        populations = boltzmann / boltzmann.sum()

    weights_ab = populations[:, None] * np.abs(M_eig) ** 2
    energy_ab = evals[:, None] - evals[None, :]
    delays = np.asarray(delta_t_array, dtype=float)
    c_t0 = np.array([
        np.sum(weights_ab * np.exp(1j * energy_ab * delay)) for delay in delays
    ])
    c_0t = np.array([
        np.sum(weights_ab * np.exp(-1j * energy_ab * delay)) for delay in delays
    ])
    return c_t0, c_0t


def spectral_D_M(
    n_spins, J, delta, temperature, delta_t_array,
    probe_model="local_gaussian", probe_sigma_sites=1.0,
    h_z=0.0, periodic=False,
):
    """Return ``D_M(t)=1/2 <(M(t)-M(0))^2>`` for a stationary state."""
    c_t0, c_0t = spectral_magnetization_correlators(
        n_spins, J, delta, temperature, delta_t_array,
        probe_model, probe_sigma_sites, h_z=h_z, periodic=periodic,
    )
    equal_time = c_t0[0].real if np.asarray(delta_t_array)[0] == 0 else None
    if equal_time is None:
        c_equal, _ = spectral_magnetization_correlators(
            n_spins, J, delta, temperature, [0.0],
            probe_model, probe_sigma_sites, h_z=h_z, periodic=periodic,
        )
        equal_time = c_equal[0].real
    return np.real(equal_time - 0.5 * (c_t0 + c_0t))
