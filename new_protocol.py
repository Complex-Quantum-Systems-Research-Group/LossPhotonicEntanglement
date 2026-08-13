"""Sequential two-photon EP-MOKS protocol.

Primary model:
  1. prepare a polarization Bell pair and a thermal XXZ spin state;
  2. photon 1 receives an impulsive magnetization-conditioned Kerr rotation;
  3. spins evolve freely for delay Delta t;
  4. photon 2 receives the same type of Kerr rotation;
  5. trace out spins and analyze the two-photon polarization state.

The impulsive approximation avoids interpreting bosonic occupation entanglement
as polarization entanglement.  ``build_protocol_components`` exposes the
Hamiltonian, generators, and unitaries so the pre-run validation can test the
same operators that the production sweep actually uses.
"""
from __future__ import annotations

import numpy as np

from hamiltonians import (
    build_spin_hamiltonian_xxz,
    exchange_interaction_generator,
    kerr_interaction_generator,
    probe_weights_gaussian,
    weighted_magnetization_z,
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


def build_protocol_components(
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
) -> dict:
    """Build the exact state/operators/unitaries used by one protocol point.

    Returning these components makes the production path inspectable: validation
    can directly test Hermiticity, commutators, and U^dagger U without rebuilding
    a subtly different protocol.
    """
    Hs = build_spin_hamiltonian_xxz(n_spins, J, delta, h_z=h_z, periodic=periodic)
    rho_s = thermal_state_from_hamiltonian(Hs, temperature)
    rho_p = bell_polarization_state(bell_state)
    rho0 = np.kron(rho_p, rho_s)

    weights = build_probe_weights(n_spins, probe_model, probe_sigma_sites)
    probe_operator = None

    if interaction_type == "kerr":
        probe_operator = weighted_magnetization_z(n_spins, weights)
        G1 = kerr_interaction_generator(n_spins, 0, probe_operator)
        G2 = kerr_interaction_generator(n_spins, 1, probe_operator)
    elif interaction_type == "exchange_benchmark":
        G1 = exchange_interaction_generator(n_spins, 0, weights)
        G2 = exchange_interaction_generator(n_spins, 1, weights)
    else:
        raise ValueError(f"unknown interaction_type={interaction_type!r}")

    U1 = unitary_from_generator(G1, theta1)
    Udelay = unitary_from_hamiltonian(embed_spin_only(Hs), delta_t)
    U2 = unitary_from_generator(G2, theta2)

    return {
        "Hs": Hs,
        "rho_spin": rho_s,
        "rho_photons": rho_p,
        "rho_initial": rho0,
        "weights": weights,
        "probe_operator": probe_operator,
        "G1": G1,
        "G2": G2,
        "U1": U1,
        "Udelay": Udelay,
        "U2": U2,
        "unitaries": (U1, Udelay, U2),
    }


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
    components = build_protocol_components(
        n_spins=n_spins,
        J=J,
        delta=delta,
        temperature=temperature,
        delta_t=delta_t,
        theta1=theta1,
        theta2=theta2,
        probe_model=probe_model,
        probe_sigma_sites=probe_sigma_sites,
        h_z=h_z,
        periodic=periodic,
        interaction_type=interaction_type,
        bell_state=bell_state,
    )
    return apply_unitaries(components["rho_initial"], components["unitaries"])
