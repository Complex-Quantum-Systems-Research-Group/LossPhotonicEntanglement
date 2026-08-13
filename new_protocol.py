"""Sequential two-photon EP-MOKS protocol.

Primary model:
  1. prepare a polarization Bell pair and a thermal XXZ spin state;
  2. photon 1 receives an impulsive magnetization-conditioned Kerr rotation;
  3. spins evolve freely for delay Delta t;
  4. photon 2 receives the same type of Kerr rotation;
  5. trace out spins and analyze the two-photon polarization state.

Two independent, separately-verified performance paths exist:

- build_protocol_components always builds dense U1, Udelay, U2 matrices
  (via new_evolution.kerr_rotation_unitary / unitary_from_spin_hamiltonian_
  embedded, themselves built from the exact kron-sum shortcuts). This is
  used wherever the actual dense unitary matrices are needed for inspection
  or validation (e.g. pipeline.py's unitarity checks), and is unchanged in
  interface/behavior from the original implementation -- it is fast because
  the underlying shortcuts avoid ever exponentiating the full
  (4 * 2^N)-dimensional generator, not because the matrices themselves are
  avoided.

- full_pipeline_unitary, for interaction_type="kerr", uses the cheaper
  apply_kron_sum_sequence path: it builds the kron-sum term lists directly
  (never forming U1/Udelay/U2 as dense (4*2^N)-dimensional matrices at all)
  and applies them to rho via tensor-reshape + einsum. This is the
  production hot path used by pipeline.py's sweep loop. For
  interaction_type="exchange_benchmark" (whose generator is a sum of two
  non-commuting tensor-product terms with no known kron-sum reduction),
  full_pipeline_unitary falls back to build_protocol_components + the
  generic dense apply_unitaries path, identical to the original
  implementation.

Both paths were verified to agree with each other and with the fully
generic dense-expm/dense-multiply implementation to machine precision
(~1e-16) at small system sizes before being deployed here.
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
from new_evolution import (
    apply_kron_sum_sequence,
    apply_unitaries,
    delay_terms,
    kerr_rotation_terms,
    kerr_rotation_unitary,
    unitary_from_generator,
    unitary_from_hamiltonian,
    unitary_from_spin_hamiltonian_embedded,
)
from operators import embed_spin_only
from states import bell_polarization_state, thermal_state_from_hamiltonian


def build_probe_weights(n_spins, probe_model, probe_sigma_sites=1.0):
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
    n_spins, J, delta, temperature, delta_t, theta1, theta2,
    probe_model="local_gaussian", probe_sigma_sites=1.0, h_z=0.0,
    periodic=False, interaction_type="kerr", bell_state="phi_plus",
):
    """Build the exact state/operators/dense unitaries used by one protocol
    point. Returning these components makes the production path inspectable:
    validation can directly test Hermiticity, commutators, and U^dagger U
    without rebuilding a subtly different protocol.

    Always returns dense U1, Udelay, U2 matrices (used by pipeline.py's
    unitarity checks); see module docstring re: full_pipeline_unitary's
    separate cheaper apply-only path for the production sweep.
    """
    Hs = build_spin_hamiltonian_xxz(n_spins, J, delta, h_z=h_z, periodic=periodic)
    rho_s = thermal_state_from_hamiltonian(Hs, temperature)
    rho_p = bell_polarization_state(bell_state)
    rho0 = np.kron(rho_p, rho_s)

    weights = build_probe_weights(n_spins, probe_model, probe_sigma_sites)
    probe_operator = None

    if interaction_type == "kerr":
        probe_operator = weighted_magnetization_z(n_spins, weights)
        # G1, G2 are cheap to form (a single kron_all call, O(dim^2)); only
        # the exponentiation was expensive, so they are still built and
        # returned for inspection/testing.
        G1 = kerr_interaction_generator(n_spins, 0, probe_operator)
        G2 = kerr_interaction_generator(n_spins, 1, probe_operator)
        U1 = kerr_rotation_unitary(theta1, 0, probe_operator)
        U2 = kerr_rotation_unitary(theta2, 1, probe_operator)
    elif interaction_type == "exchange_benchmark":
        G1 = exchange_interaction_generator(n_spins, 0, weights)
        G2 = exchange_interaction_generator(n_spins, 1, weights)
        U1 = unitary_from_generator(G1, theta1)
        U2 = unitary_from_generator(G2, theta2)
    else:
        raise ValueError(f"unknown interaction_type={interaction_type!r}")

    Udelay = unitary_from_spin_hamiltonian_embedded(Hs, delta_t)

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
    n_spins, J, delta, temperature, delta_t, theta1, theta2,
    probe_model="local_gaussian", probe_sigma_sites=1.0, h_z=0.0,
    periodic=False, interaction_type="kerr", bell_state="phi_plus",
):
    """Return the final full density matrix after the sequential protocol.

    For interaction_type="kerr" (the primary model and production sweep
    path), uses the cheap kron-sum apply path (never forms the dense
    (4*2^N)-dimensional U1/Udelay/U2 matrices). For
    interaction_type="exchange_benchmark", falls back to
    build_protocol_components + the generic dense apply_unitaries path.
    """
    Hs = build_spin_hamiltonian_xxz(n_spins, J, delta, h_z=h_z, periodic=periodic)
    rho_s = thermal_state_from_hamiltonian(Hs, temperature)
    rho_p = bell_polarization_state(bell_state)
    rho0 = np.kron(rho_p, rho_s)

    weights = build_probe_weights(n_spins, probe_model, probe_sigma_sites)

    if interaction_type == "kerr":
        probe_operator = weighted_magnetization_z(n_spins, weights)
        terms_sequence = (
            kerr_rotation_terms(theta1, 0, probe_operator),
            delay_terms(Hs, delta_t),
            kerr_rotation_terms(theta2, 1, probe_operator),
        )
        return apply_kron_sum_sequence(rho0, terms_sequence)

    # interaction_type == "exchange_benchmark" (validated in
    # build_protocol_components; invalid types raise ValueError there)
    components = build_protocol_components(
        n_spins=n_spins, J=J, delta=delta, temperature=temperature,
        delta_t=delta_t, theta1=theta1, theta2=theta2, probe_model=probe_model,
        probe_sigma_sites=probe_sigma_sites, h_z=h_z, periodic=periodic,
        interaction_type=interaction_type, bell_state=bell_state,
    )
    return apply_unitaries(components["rho_initial"], components["unitaries"])