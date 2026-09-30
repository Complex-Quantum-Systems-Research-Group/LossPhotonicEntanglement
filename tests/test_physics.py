import numpy as np

from hamiltonians import (
    build_spin_hamiltonian_xxz,
    collective_magnetization_z,
    probe_weights_gaussian,
    weighted_magnetization_z,
)
from measures import bell_fidelity, concurrence, l1_coherence, mutual_information, purity, von_neumann_entropy
from new_protocol import build_protocol_components, full_pipeline_unitary
from observables import partial_trace_spins
from states import bell_polarization_state, thermal_state_from_hamiltonian
from validation import assert_density_matrix, assert_unitary
from operators import SX, SY, spin_only_operator
from correlations import spectral_D_M, spectral_magnetization_correlators
from channel_diagnostics import channel_residuals, reconstruct_choi, validate_choi_reconstruction
from parity_preflight import PHI_PLUS, PSI_MINUS, analytic_consequences, spin_characteristic_C
from sector_correlations import (
    diagonalize_magnetization_sectors,
    infinite_temperature_M2,
    sector_D_M,
)


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


def test_spectral_correlators_match_brute_force_heisenberg_picture():
    """The exported ordered correlators and D_M must match direct evolution."""
    from scipy.linalg import expm

    n, J, delta, temperature = 4, 1.0, 1.2, 0.3
    delays = np.array([0.0, 0.7, 1.7])
    H = build_spin_hamiltonian_xxz(n, J, delta)
    rho = thermal_state_from_hamiltonian(H, temperature)
    M = weighted_magnetization_z(n, probe_weights_gaussian(n, sigma=0.8))

    expected_t0, expected_0t, expected_d = [], [], []
    for delay in delays:
        U = expm(-1j * H * delay)
        M_t = U.conj().T @ M @ U
        c_t0 = np.trace(rho @ M_t @ M)
        c_0t = np.trace(rho @ M @ M_t)
        expected_t0.append(c_t0)
        expected_0t.append(c_0t)
        expected_d.append(0.5 * np.trace(rho @ (M_t - M) @ (M_t - M)))

    actual_t0, actual_0t = spectral_magnetization_correlators(
        n, J, delta, temperature, delays, "local_gaussian", 0.8
    )
    actual_d = spectral_D_M(n, J, delta, temperature, delays, "local_gaussian", 0.8)
    assert np.allclose(actual_t0, expected_t0, atol=1e-12, rtol=0.0)
    assert np.allclose(actual_0t, expected_0t, atol=1e-12, rtol=0.0)
    assert np.allclose(actual_d, expected_d, atol=1e-12, rtol=0.0)


def test_equal_coupling_weak_law_has_theta_four_residual():
    """For Phi+ and equal coupling, halving theta reduces residual by ~16."""
    n, J, delta, temperature, delay = 4, 1.0, 1.0, 0.3, 1.7
    D_M = spectral_D_M(
        n, J, delta, temperature, [delay], "local_gaussian", 0.8
    )[0]
    target = bell_polarization_state()
    residuals = []
    for theta in (0.2, 0.1, 0.05):
        rho_full = full_pipeline_unitary(
            n_spins=n, J=J, delta=delta, temperature=temperature,
            delta_t=delay, theta1=theta, theta2=theta,
            probe_model="local_gaussian", probe_sigma_sites=0.8,
        )
        simulated = bell_fidelity(partial_trace_spins(rho_full, n), target)
        predicted = 1.0 - 2.0 * theta**2 * D_M
        residuals.append(abs(simulated - predicted))

    ratios = np.array(residuals[:-1]) / np.array(residuals[1:])
    assert np.all((ratios > 14.0) & (ratios < 18.0)), (residuals, ratios)


def test_reconstructed_primary_channel_is_cptp_and_unital():
    """A full 16-matrix-unit reconstruction supports channel-level claims."""
    choi = reconstruct_choi(
        n_spins=3, J=1.0, delta=1.1, temperature=0.4, delta_t=0.8,
        theta1=0.31, theta2=0.19, probe_model="local_gaussian",
        probe_sigma_sites=0.8,
    )
    diagnostics = channel_residuals(choi)
    assert diagnostics["choi_min_eigenvalue"] > -1e-12
    assert diagnostics["hermiticity_residual"] < 1e-12
    assert diagnostics["trace_preservation_residual"] < 1e-12
    assert diagnostics["unitality_residual"] < 1e-12
    report = validate_choi_reconstruction(choi)
    assert report["cptp_pass"]
    assert "unitality_residual" not in {
        key: value for key, value in report.items() if key != "validation"
    }
    assert "Tr_out(J) = I" in report["choi_convention"]
    assert "J[(i,a),(j,b)]" in report["choi_convention"]


def test_full_pipeline_accepts_nonhermitian_matrix_unit():
    """Choi matrix units use the production path without density assertions."""
    matrix_unit = np.zeros((4, 4), dtype=complex)
    matrix_unit[0, 1] = 1.0
    output = full_pipeline_unitary(
        n_spins=3, J=1.0, delta=1.1, temperature=0.4, delta_t=0.8,
        theta1=0.31, theta2=0.19, probe_model="local_gaussian",
        probe_sigma_sites=0.8, photon_operator=matrix_unit,
    )
    assert output.shape == (32, 32)
    assert not np.allclose(output, output.conj().T)


def test_nonperturbative_plot_path_skips_weak_residual_gate(tmp_path, monkeypatch, capsys):
    import plot_results

    rows = [
        {
            "T": 0.3, "T_kelvin": 100.0, "delta_t": delay,
            "delta_t_fs": delay * 19.0, "bell_fidelity": fidelity,
            "concurrence": concurrence_value, "n_spins": 4, "J": 1.0,
            "delta": 1.0, "h_z": 0.0, "periodic": False,
            "theta1": 0.4, "theta2": 0.4, "probe_model": "local_gaussian",
            "probe_sigma_sites": 0.8, "bell_state": "phi_plus",
        }
        for delay, fidelity, concurrence_value in (
            (0.0, 1.0, 1.0), (1.0, 0.98, 0.96)
        )
    ]
    monkeypatch.setattr(
        plot_results, "compute_D_M",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("residual comparison must not run")
        ),
    )
    monkeypatch.setattr(plot_results, "export_correlators", lambda *args: None)
    plot_results.run_weak_coupling_check(rows, tmp_path, tmp_path / "summary.json")
    output = capsys.readouterr().out
    assert "SKIP weak-coupling check" in output
    assert not (tmp_path / "weak_coupling_residual.png").exists()


def test_campaign_namespaces_figure_directory(tmp_path):
    import plot_results

    summary = tmp_path / "data" / "summary.json"
    assert plot_results._figs_dir(summary, campaign="smoke_theta04") == (
        tmp_path / "figs" / "smoke_theta04"
    )


def test_analyzer_nonperturbative_gate_is_a_skip(capsys):
    import analyze_results

    analyze_results._report_weak_coupling_gate([
        {"theta1": 0.4, "theta2": 0.4, "campaign": "smoke_theta04"}
    ])
    assert "SKIP weak-coupling gate" in capsys.readouterr().out


def test_reported_weak_coupling_exponent_is_four():
    import plot_results

    rows = [
        {"T": 0.3, "T_kelvin": 100.0, "delta_t": 0.0},
        {"T": 0.3, "T_kelvin": 100.0, "delta_t": 1.7},
    ]
    metadata = {
        "n_spins": 4, "J": 1.0, "delta": 1.0, "h_z": 0.0,
        "periodic": False, "probe_model": "local_gaussian",
        "probe_sigma_sites": 0.8, "bell_state": "phi_plus",
    }
    fit = plot_results.fit_weak_coupling_exponent(metadata, {100.0: rows})
    assert 3.9 < fit["exponent"] < 4.1


def test_spin_characteristic_is_real_when_spin_flip_parity_holds():
    """Prop. 6b preflight: h_z=0 preserves global x-spin-flip parity."""
    for temperature in (0.05, 0.76):
        for delay in (0.7, 2.3, 4.0):
            C = spin_characteristic_C(
                n_spins=6, J=1.0, delta=1.0, h_z=0.0,
                temperature=temperature, delay=delay,
                theta1=0.4, theta2=0.4,
            )
            assert abs(C.imag) < 1e-12


def test_longitudinal_field_breaks_bell_diagonality_collapse():
    """A longitudinal field breaks R symmetry and opens the Prop. 6 gap."""
    n = 6
    kwargs = dict(
        n_spins=n, J=1.0, delta=1.0, h_z=0.5,
        temperature=0.05, delta_t=3.0, theta1=0.4, theta2=0.4,
        probe_model="local_gaussian", probe_sigma_sites=1.0,
    )
    C = spin_characteristic_C(
        delay=kwargs["delta_t"],
        **{k: v for k, v in kwargs.items() if k != "delta_t"},
    )
    rho_p = partial_trace_spins(full_pipeline_unitary(**kwargs), n)
    bell_cross = np.vdot(PHI_PLUS, rho_p @ PSI_MINUS)

    assert abs(C.imag) > 1e-5
    assert l1_coherence(rho_p) > 1.0001
    assert abs(bell_cross) > 1e-5


def test_parity_identity_and_unconditional_rank_two_support():
    """Exact magnitudes distinguish Prop. 6a support from 6b diagonality."""
    n = 6
    for h_z in (0.0, 0.1, 0.5):
        kwargs = dict(
            n_spins=n, J=1.0, delta=1.0, h_z=h_z,
            temperature=0.76, delta_t=3.0, theta1=0.4, theta2=0.4,
            probe_model="local_gaussian", probe_sigma_sites=1.0,
        )
        C = spin_characteristic_C(
            delay=kwargs["delta_t"],
            **{k: v for k, v in kwargs.items() if k != "delta_t"},
        )
        p_a, coh_a, cl1_a = analytic_consequences(C)
        rho_p = partial_trace_spins(full_pipeline_unitary(**kwargs), n)
        fidelity = float(np.vdot(PHI_PLUS, rho_p @ PHI_PLUS).real)
        bell_cross = float(abs(np.vdot(PHI_PLUS, rho_p @ PSI_MINUS)))

        assert abs(fidelity - (1.0 - p_a)) < 1e-12
        assert abs(bell_cross - coh_a) < 1e-12
        assert abs(l1_coherence(rho_p) - cl1_a) < 1e-12
        assert np.count_nonzero(np.linalg.eigvalsh(rho_p) > 1e-12) <= 2
        if h_z == 0.0:
            assert abs(C.imag) < 1e-14
        else:
            assert abs(C.imag) > 1e-6


def test_sector_D_M_matches_full_spin_diagonalization_and_parity_pairing():
    n = 6
    delays = np.arange(0.0, 6.1, 0.3)
    half, weights = diagonalize_magnetization_sectors(
        n, 1.0, 1.0, "local_gaussian", 1.0, use_spin_flip=True
    )
    full, _ = diagonalize_magnetization_sectors(
        n, 1.0, 1.0, "local_gaussian", 1.0, use_spin_flip=False
    )
    for temperature in (0.0152, 1.2673):
        dm_half, _ = sector_D_M(half, temperature, delays, relative_cutoff=0.0)
        dm_full, _ = sector_D_M(full, temperature, delays, relative_cutoff=0.0)
        dm_reference = spectral_D_M(
            n, 1.0, 1.0, temperature, delays, "local_gaussian", 1.0
        )
        assert np.max(np.abs(dm_half - dm_full)) < 1e-12
        assert np.max(np.abs(dm_half - dm_reference)) < 1e-12
        assert dm_half[0] == 0.0

    _, high_meta = sector_D_M(half, 1e12, [0.0], relative_cutoff=0.0)
    assert abs(high_meta["S0"] - infinite_temperature_M2(weights)) < 1e-12
