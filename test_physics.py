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


def test_equal_coupling_zero_delay_invariance_any_probe():
    """Proposition 4: theta1=theta2=theta, dt=0 => rho_P = rho_Bell for any
    probe profile (nonuniform included), not only the conserved collective one.
    """
    target = bell_polarization_state()
    theta = 0.41  # arbitrary nonzero angle
    for probe_model in ("local_gaussian", "single_site", "collective"):
        for T in (0.0, 0.5, 1.8):
            rho_full = full_pipeline_unitary(
                n_spins=4,
                J=-1.0,
                delta=1.2,
                temperature=T,
                delta_t=0.0,
                theta1=theta,
                theta2=theta,
                probe_model=probe_model,
                probe_sigma_sites=0.8,
            )
            rho_p = partial_trace_spins(rho_full, 4)
            assert np.linalg.norm(rho_p - target) < 1e-10, (
                f"probe_model={probe_model}, T={T}"
            )


def test_equal_coupling_zero_delay_invariance_fails_for_unequal_theta():
    """Sanity check that the invariance is specific to theta1=theta2 at dt=0,
    i.e. the test above is not vacuously true for any theta1, theta2.
    """
    target = bell_polarization_state()
    rho_full = full_pipeline_unitary(
        n_spins=4,
        J=-1.0,
        delta=1.2,
        temperature=0.5,
        delta_t=0.0,
        theta1=0.41,
        theta2=0.17,
        probe_model="local_gaussian",
        probe_sigma_sites=0.8,
    )
    rho_p = partial_trace_spins(rho_full, 4)
    assert np.linalg.norm(rho_p - target) > 1e-6


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


def _brute_force_reference_state(
    n_spins, J, delta, temperature, delta_t, theta1, theta2,
    probe_model, probe_sigma_sites, interaction_type, bell_state="phi_plus",
):
    """Fully independent reference implementation of the protocol, used only
    by test_fast_path_matches_brute_force_reference below. Deliberately
    bypasses every performance shortcut in new_evolution.py/new_protocol.py
    (kerr_rotation_terms, delay_terms, apply_kron_sum, kerr_rotation_unitary,
    unitary_from_spin_hamiltonian_embedded): it builds the full generators
    via kron_all and exponentiates the full (4 * 2^n_spins)-dimensional
    operators directly with scipy.linalg.expm, then applies them via plain
    dense matrix multiplication. If this ever disagrees with
    full_pipeline_unitary, the fast path has a bug.
    """
    from scipy.linalg import expm

    from hamiltonians import (
        exchange_interaction_generator,
        kerr_interaction_generator,
    )
    from new_protocol import build_probe_weights
    from operators import embed_spin_only

    Hs = build_spin_hamiltonian_xxz(n_spins, J, delta)
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

    U1 = expm(-1j * theta1 * G1)
    Udelay = expm(-1j * delta_t * embed_spin_only(Hs))
    U2 = expm(-1j * theta2 * G2)

    out = rho0
    for U in (U1, Udelay, U2):
        out = U @ out @ U.conj().T
    return out


def test_fast_path_matches_brute_force_reference():
    """Regression test: full_pipeline_unitary's fast kron-sum path (the
    production path used by pipeline.py's sweep loop) must exactly match a
    fully independent, brute-force dense-expm implementation that bypasses
    every shortcut in new_evolution.py/new_protocol.py (see
    _brute_force_reference_state above).

    This is the guard the rest of the suite is missing: pipeline.py's own
    unitarity checks (test_protocol_unitaries_are_unitary above) only
    exercise the separate dense-construction path
    (build_protocol_components -> kerr_rotation_unitary /
    unitary_from_spin_hamiltonian_embedded), which is used for validation
    and inspection. The sweep itself calls full_pipeline_unitary, which for
    interaction_type="kerr" takes an entirely different code path
    (kerr_rotation_terms / delay_terms / apply_kron_sum, which never forms
    the dense unitaries at all). Without this test, nothing in the suite
    directly confirms the two paths agree.

    Covers: both interaction types (kerr and exchange_benchmark), three
    probe models, a theta=0 edge case, and a delta_t=0 edge case.
    """
    n = 5
    cases = [
        dict(
            n_spins=n, J=1.0, delta=1.0, temperature=0.5, delta_t=0.7,
            theta1=0.31, theta2=0.19, probe_model="local_gaussian",
            probe_sigma_sites=0.9, interaction_type="kerr",
        ),
        dict(
            n_spins=n, J=1.0, delta=1.0, temperature=0.0, delta_t=0.0,
            theta1=0.4, theta2=0.4, probe_model="single_site",
            probe_sigma_sites=1.0, interaction_type="kerr",
        ),
        dict(
            n_spins=n, J=1.0, delta=1.0, temperature=1.3, delta_t=2.1,
            theta1=0.0, theta2=0.0, probe_model="collective",
            probe_sigma_sites=1.0, interaction_type="kerr",
        ),
        dict(
            n_spins=n, J=1.0, delta=1.0, temperature=0.8, delta_t=0.6,
            theta1=0.25, theta2=0.15, probe_model="local_gaussian",
            probe_sigma_sites=0.9, interaction_type="exchange_benchmark",
        ),
    ]

    for case in cases:
        ref = _brute_force_reference_state(**case)
        fast = full_pipeline_unitary(
            n_spins=case["n_spins"],
            J=case["J"],
            delta=case["delta"],
            temperature=case["temperature"],
            delta_t=case["delta_t"],
            theta1=case["theta1"],
            theta2=case["theta2"],
            probe_model=case["probe_model"],
            probe_sigma_sites=case["probe_sigma_sites"],
            interaction_type=case["interaction_type"],
        )
        err = np.linalg.norm(ref - fast)
        assert err < 1e-10, (
            f"fast path diverged from brute-force reference: {case}, "
            f"error={err}"
        )