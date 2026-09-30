import numpy as np
import pytest
from scipy.linalg import expm

from hamiltonians import build_spin_hamiltonian_xxz, weighted_magnetization_z
from operators import I2, SY, SZ
from thermometry import (BELL, ThermometryChannel, thermal_populations,
                        quantum_fisher_information, output_from_probabilities,
                        product_probabilities, classical_fisher_information,
                        diagnose_input, optimize_inputs)


@pytest.mark.parametrize('eta,reference', [(0., (.00015227, .01656450)), (.25, (.01432292, .02238723))])
def test_independent_joint_propagation(eta, reference):
    n, temperature, delay = 3, .7, 1.7
    weights = np.array([.5, .3, .2])
    c = ThermometryChannel(n, weights, eta1=eta, eta2=eta)
    h = build_spin_hamiltonian_xxz(n, 1., 1.)
    m = weighted_magnetization_z(n, weights)
    g = .4*SY + eta*SZ
    unitary = (expm(-1j*np.kron(np.kron(I2, g), m)) @
               expm(-1j*delay*np.kron(np.eye(4), h)) @
               expm(-1j*np.kron(np.kron(g, I2), m)))
    e, v = np.linalg.eigh(h)
    w, dw, bound = thermal_populations(e, temperature)
    rng = np.random.default_rng(918)
    product = np.kron(np.array([.3, .8j])/np.sqrt(.73), np.array([.6j, .2])/np.sqrt(.4))
    entangled = rng.normal(size=4) + 1j*rng.normal(size=4)
    entangled /= np.linalg.norm(entangled)
    k, dk = c.kernel(temperature, delay)
    for vector in (BELL, product, entangled):
        a = c.basis.conj().T @ vector
        state = np.outer(a, a.conj())
        for pop, kernel in ((w, k), (dw, dk)):
            initial = np.kron(np.outer(vector, vector.conj()), (v*pop) @ v.conj().T)
            joint = unitary @ initial @ unitary.conj().T
            reduced = np.trace(joint.reshape(4, 8, 4, 8), axis1=1, axis2=3)
            np.testing.assert_allclose(reduced, c.basis @ (kernel*state) @ c.basis.conj().T, atol=2e-14)
        diagnostics = diagnose_input(c, temperature, delay, vector)
        assert diagnostics['derivative_absolute_errors'][-1] < 1e-8
        assert diagnostics['F_Q'] <= bound+1e-12
        phases = np.exp(1j*rng.uniform(-np.pi, np.pi, 4))
        positive = np.abs(a)
        r, d = k*np.outer(positive, positive), dk*np.outer(positive, positive)
        phase_matrix = np.outer(phases, phases.conj())
        assert quantum_fisher_information(r, d) == pytest.approx(quantum_fisher_information(r*phase_matrix, d*phase_matrix), abs=1e-12)
    bell_qfi = quantum_fisher_information(*output_from_probabilities(c.q_bell, k, dk))
    product_qfi = quantum_fisher_information(*output_from_probabilities(product_probabilities(.5, .5), k, dk))
    np.testing.assert_allclose([bell_qfi, product_qfi], reference, atol=5e-9, rtol=0)


def test_binary_mixture_zero_delay_and_zero_coupling():
    c = ThermometryChannel(3, [.5, .3, .2])
    for t in (0., 1.7, 4.):
        k, dk = c.kernel(.7, t)
        a = c.bell_control
        r, d = [c.basis @ (z*np.outer(a, a.conj())) @ c.basis.conj().T for z in (k, dk)]
        p, dp = np.vdot(BELL, r @ BELL).real, np.vdot(BELL, d @ BELL).real
        fq = quantum_fisher_information(r, d)
        if t == 0:
            assert fq < 1e-24
            assert np.linalg.norm(d) < 1e-14
        else:
            assert 0 < p < 1
            assert fq == pytest.approx(dp**2/(p*(1-p)), abs=1e-12)
    zero = ThermometryChannel(3, [.5, .3, .2], theta1=0., theta2=0.)
    k, dk = zero.kernel(.7, 1.7)
    assert np.linalg.norm(dk) < 1e-14
    assert quantum_fisher_information(*output_from_probabilities(np.full(4, .25), k, dk)) < 1e-24


def test_optimizer_and_budget():
    c = ThermometryChannel(3, [.5, .3, .2], eta1=.25, eta2=.25)
    k, dk = c.kernel(.7, 1.7)
    result = optimize_inputs(k, dk, c.q_bell)
    assert result['F_separable_found'] >= .02238722
    assert result['F_all_inputs_found'] >= max(result['F_separable_found'], result['F_bell'])
    rho = np.diag([.2, .3, .1, .4])
    drho = np.diag([.1, -.1, .2, -.2])
    assert classical_fisher_information(rho, drho) <= quantum_fisher_information(rho, drho)
    with pytest.raises(ValueError, match='budget'):
        classical_fisher_information(rho, drho, fractions=np.ones(9))
    with pytest.raises(ValueError, match='trace'):
        quantum_fisher_information(rho, drho+np.eye(4))
    with pytest.raises(ValueError, match='positive'):
        thermal_populations(np.array([0., 1.]), 0.)


def test_asymmetric_controls_field_and_periodic_sectors():
    # Separately check the sector spectra outside the parity-symmetric defaults.
    c = ThermometryChannel(3, [.5, -.3, .2], theta1=.2, theta2=.4,
                          eta1=-.1, eta2=.25, delta=.8, h_over_J=.13, periodic=True)
    h = build_spin_hamiltonian_xxz(3, 1., .8, h_z=.13, periodic=True)
    np.testing.assert_allclose(np.sort(c.energies), np.linalg.eigvalsh(h), atol=1e-14)
    result = diagnose_input(c, .7, 1.7, BELL)
    assert result['derivative_absolute_errors'][-1] < 1e-8


def test_delay_candidate_is_retained_if_fresh_search_misses_it(monkeypatch):
    import thermometry_benchmark as runner
    c = ThermometryChannel(3, [.5, .3, .2])
    def failed_search(*args, **kwargs):
        return dict(F_bell=.00015227, F_separable_found=0., F_all_inputs_found=0.,
                    u_opt=0., v_opt=0., q_all_opt=[1., 0., 0., 0.], optimizer_seed=[17],
                    optimizer_status=[{'method': 'product_DE', 'value': 0., 'success': False}])
    monkeypatch.setattr(runner, 'optimize_inputs', failed_search)
    row = runner.evaluate(c, .7, 1.7, dict(theta1=.4, theta2=.4, eta1=0., eta2=0.), (17,), retained_product=(.5, .5))
    assert row['F_separable_found'] == pytest.approx(.01656450, abs=5e-9)
    assert row['F_all_inputs_found'] >= row['F_separable_found']
    assert row['u_opt'] == row['v_opt'] == .5
