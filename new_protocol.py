"""Sequential two-photon EP-MOKS protocol.

Primary model:
  1. prepare a polarization Bell pair and a thermal XXZ spin state;
  2. photon 1 receives an impulsive magnetization-conditioned Kerr rotation;
  3. spins evolve freely for delay Delta t;
  4. photon 2 receives the same type of Kerr rotation;
  5. trace out spins and analyze the two-photon polarization state.

The impulsive approximation removes the artificial final free-evolution stage and
avoids interpreting bosonic occupation entanglement as polarization entanglement.
"""
from __future__ import annotations

import numpy as np

from hamiltonians import (
    build_spin_hamiltonian_xxz,
    probe_weights_gaussian,
    weighted_magnetization_z,
    kerr_interaction_generator,
    exchange_interaction_generator,
)
from new_evolution import apply_unitaries, unitary_from_generator, unitary_from_hamiltonian
from operators import embed_spin_only
from states import bell_polarization_state, thermal_state_from_hamiltonian


def build_probe_weights(n_spins: int, probe_model: str, probe_sigma_sites: float = 1.0) -> np.ndarray:
    if probe_model == "local_gaussian":
        return probe_weights_gaussian(n_spins, sigma=probe_sigma_sites)
    if probe_model == "collective":
        return np.ones(n_spins, dtype=float) / n_spins
    if probe_model == "single_site":
        w = np.zeros(n_spins, dtype=float)
        w[n_spins // 2] = 1.0
        return w
    raise ValueError(f"unknown probe_model={probe_model!r}")


def full_pipeline_unitary(
    n_spins: int,
    J: float,
    delta: float,
    temperature: float,
    delta_t: float,
    theta1: float,
    theta2: float,
    probe_model: str = "local_gaussian",
    probe_sigma_sites: float = 1.0,
    h_z: float = 0.0,
    periodic: bool = False,
    interaction_type: str = "kerr",
    bell_state: str = "phi_plus",
) -> np.ndarray:
    """Return the final full density matrix after the sequential protocol."""
    Hs = build_spin_hamiltonian_xxz(n_spins, J, delta, h_z=h_z, periodic=periodic)
    rho_s = thermal_state_from_hamiltonian(Hs, temperature)
    rho_p = bell_polarization_state(bell_state)
    rho0 = np.kron(rho_p, rho_s)

    weights = build_probe_weights(n_spins, probe_model, probe_sigma_sites)

    if interaction_type == "kerr":
        M = weighted_magnetization_z(n_spins, weights)
        G1 = kerr_interaction_generator(n_spins, 0, M)
        G2 = kerr_interaction_generator(n_spins, 1, M)
    elif interaction_type == "exchange_benchmark":
        G1 = exchange_interaction_generator(n_spins, 0, weights)
        G2 = exchange_interaction_generator(n_spins, 1, weights)
    else:
        raise ValueError(f"unknown interaction_type={interaction_type!r}")

    U1 = unitary_from_generator(G1, theta1)
    Udelay = unitary_from_hamiltonian(embed_spin_only(Hs), delta_t)
    U2 = unitary_from_generator(G2, theta2)
    return apply_unitaries(rho0, [U1, Udelay, U2])
