import numpy as np

from hamiltonians import (
    build_spin_hamiltonian_xxz,
    collective_magnetization_z,
    probe_weights_gaussian,
    weighted_magnetization_z,
)
from measures import concurrence, mutual_information, purity, von_neumann_entropy
from new_protocol import build_protocol_components, full_pipeline_unitary
from observables import partial_trace_spins
from states import bell_polarization_state, thermal_state_from_hamiltonian
from validation import assert_density_matrix, assert_unitary
from operators import SX, SY, spin_only_operator


def test_bell_state_baseline():
    rho = bell_polarization_state()
    assert_density_matrix(rho)
    assert abs(concurrence(rho) - 1.0) < 1e-12
    assert abs(mutual_information(rho) - 2.0) < 1e-12
    assert abs(purity(rho) - 1.0) < 1e-12
    assert abs(von_neumann_entropy(rho)) < 1e-12


def test_thermal_state_is_physical_at_zero_and_finite_temperature():
    H = build_spin_hamiltonian_xxz(4, J=-1.0, delta=1.5)
    for T in (0.0, 0.2, 2.0):
        assert_density_matrix(thermal_state_from_hamiltonian(H, T))


def test_collective_magnetization_commutes_with_xxz():
    n = 5
    H = build_spin_hamiltonian_xxz(n, J=-1.0, delta=1.2)
    M = collective_magnetization_z(n)
    assert np.linalg.norm(H @ M - M @ H) < 1e-12


def test_local_weighted_magnetization_is_generically_not_conserved():
    n = 5
    H = build_spin_hamiltonian_xxz(n, J=-1.0, delta=1.2)
    M = weighted_magnetization_z(n, probe_weights_gaussian(n, sigma=0.8))
    assert np.linalg.norm(H @ M - M @ H) > 1e-6


def test_zero_coupling_leaves_bell_pair_unchanged_for_multiple_T_and_delay():
    target = bell_polarization_state()
    for T in (0.0, 0.7, 2.0):
        for dt in (0.0, 0.4, 1.1):
            rho_full = full_pipeline_unitary(
                n_spins=4,
                J=-1.0,
                delta=1.2,
                temperature=T,
                delta_t=dt,
                theta1=0.0,
                theta2=0.0,
                probe_model="local_gaussian",
            )
            rho_p = partial_trace_spins(rho_full, 4)
            assert np.linalg.norm(rho_p - target) < 1e-11


def test_protocol_unitaries_are_unitary():
    components = build_protocol_components(
        n_spins=4,
        J=-1.0,
        delta=1.2,
        temperature=0.7,
        delta_t=1.1,
        theta1=0.13,
        theta2=0.09,
        probe_model="local_gaussian",
    )
    assert_unitary(components["U1"])
    assert_unitary(components["Udelay"])
    assert_unitary(components["U2"])


def test_collective_probe_is_delay_independent():
    kwargs = dict(
        n_spins=4,
        J=-1.0,
        delta=1.2,
        temperature=0.7,
        theta1=0.13,
        theta2=0.09,
        probe_model="collective",
    )

    r0 = partial_trace_spins(
        full_pipeline_unitary(delta_t=0.0, **kwargs), 4
    )
    r1 = partial_trace_spins(
        full_pipeline_unitary(delta_t=1.337, **kwargs), 4
    )

    assert np.linalg.norm(r0 - r1) < 1e-10


def test_weighted_magnetization_commutator_identity():
    n = 5
    J = -1.0
    delta = 1.2

    H = build_spin_hamiltonian_xxz(n, J=J, delta=delta)
    weights = probe_weights_gaussian(n, sigma=0.8)
    M = weighted_magnetization_z(n, weights)

    lhs = H @ M - M @ H

    rhs = np.zeros_like(H, dtype=complex)

    for i in range(n - 1):
        sxi = spin_only_operator(SX, i, n)
        syi = spin_only_operator(SY, i, n)
        sxj = spin_only_operator(SX, i + 1, n)
        syj = spin_only_operator(SY, i + 1, n)

        current_pauli = sxi @ syj - syi @ sxj

        rhs += (
            0.5j
            * J
            * (weights[i] - weights[i + 1])
            * current_pauli
        )

    assert np.linalg.norm(lhs - rhs) < 1e-12