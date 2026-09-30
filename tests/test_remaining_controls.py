"""Manuscript controls using repository thermal states and Gaussian probes."""
import numpy as np
import pytest
from scipy.linalg import expm

from ellipticity_preflight import zero_delay_bell_fidelity
from new_protocol import build_protocol_components, full_pipeline_unitary, build_probe_weights
from observables import partial_trace_spins
from thermometry import ThermometryChannel, quantum_fisher_information, output_from_probabilities


def options(n=4, delay=1.7):
    return dict(n_spins=n, J=1., delta=1., temperature=.7, delta_t=delay,
                theta1=.4, theta2=.4, eta1=.25, eta2=.25,
                probe_model='local_gaussian', probe_sigma_sites=1., h_z=0., periodic=False)


def test_unequal_elliptical_unitality_n4():
    kw = options()
    kw.update(theta1=.35, eta1=.22, theta2=.28, eta2=.11)
    out = partial_trace_spins(full_pipeline_unitary(**kw, photon_operator=np.eye(4)), 4)
    assert np.linalg.norm(out-np.eye(4)) < 1e-13


def test_elliptical_zero_delay_closed_form_n4():
    kw = options(delay=0.)
    c = build_protocol_components(**kw)
    out = partial_trace_spins(full_pipeline_unitary(**kw), 4)
    expected = zero_delay_bell_fidelity(.4, .25, .4, .25, c['probe_operator'], c['rho_spin'])
    assert abs(np.trace(c['rho_photons'] @ out).real-expected) < 1e-10


def comm(a, b):
    return a @ b-b @ a


def test_second_order_matrix_and_psi_plus_population_n3():
    kw = options(n=3)
    kw.update(theta1=1., theta2=1., eta1=.6, eta2=.6)
    c = build_protocol_components(**kw)
    # Interaction picture puts the second kick at the delayed magnetization.
    w = np.kron(np.eye(4), expm(-1j*kw['delta_t']*c['Hs']))
    g1, g2 = c['G1'], w.conj().T @ c['G2'] @ w
    rho = c['rho_initial']
    second = partial_trace_spins(-.5*comm(g1, comm(g1, rho))
        -.5*comm(g2, comm(g2, rho))-comm(g2, comm(g1, rho)), 3)
    phi = np.array([1, 0, 0, 1])/np.sqrt(2)
    psi = np.array([0, 1, 1, 0])/np.sqrt(2)
    epsilons = np.array([.02, .01, .005])
    residuals, populations = [], []
    for eps in epsilons:
        run = dict(kw, theta1=eps, theta2=eps, eta1=.6*eps, eta2=.6*eps)
        out = partial_trace_spins(full_pipeline_unitary(**run), 3)
        residuals.append(np.linalg.norm(out-c['rho_photons']-eps**2*second))
        population = float(np.vdot(psi, out @ psi).real)
        coherence = abs(np.vdot(phi, out @ psi))**2
        assert population >= coherence-1e-15
        populations.append(population/eps**4)
    exponent = np.polyfit(np.log(epsilons), np.log(residuals), 1)[0]
    assert exponent == pytest.approx(4., abs=.05)
    assert min(populations) > 0
    np.testing.assert_allclose(populations, populations[-1], rtol=5e-4, atol=0)


@pytest.mark.parametrize('n', [6, 10])
@pytest.mark.parametrize('eta', [0., .25])
def test_two_sided_kernel_parity_and_production_smoke(n, eta):
    for field in (0., .5):
        channel = ThermometryChannel(n, build_probe_weights(n, 'local_gaussian', 1.),
                                     eta1=eta, eta2=eta, h_over_J=field)
        k, dk = channel.kernel(.7, 1.7)
        ratios = [np.linalg.norm(m-m[::-1, ::-1])/np.linalg.norm(m) for m in (k, dk)]
        if field == 0:
            assert max(ratios) < 1e-13
        else:
            assert min(ratios) > .1


@pytest.mark.parametrize('eta', [0., .25])
def test_kraus_symmetrization_random_inputs(eta):
    channel = ThermometryChannel(6, build_probe_weights(6, 'local_gaussian', 1.),
                                 eta1=eta, eta2=eta)
    k, dk = channel.kernel(.7, 1.7)
    rng = np.random.default_rng(1907)
    for q in rng.dirichlet(np.ones(4), size=100):
        sym = (q+q[::-1])/2
        f = lambda p: quantum_fisher_information(*output_from_probabilities(p, k, dk))
        assert f(sym) >= f(q)-1e-12


@pytest.mark.parametrize('eta', [0., .25])
def test_rational_parity_blocks_against_spectral_qfi(eta):
    from symmetric_qfi import rational_coefficients, stationary_candidates
    from remaining_computations import scores, symmetric
    from scipy.optimize import minimize_scalar
    channel = ThermometryChannel(6, build_probe_weights(6, 'local_gaussian', 1.),
                                 eta1=eta, eta2=eta)
    k, dk = channel.kernel(.7, 1.7)
    p, q = rational_coefficients(k, dk)
    ws = np.linspace(.01, .99, 31)
    values = [quantum_fisher_information(*output_from_probabilities(symmetric(w), k, dk)) for w in ws]
    np.testing.assert_allclose(p(ws)/q(ws), values, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(scores(k, dk, symmetric(ws)), values, atol=1e-14, rtol=1e-12)
    candidates = stationary_candidates(k, dk)
    best = max(scores(k, dk, symmetric(candidates)))
    opt = minimize_scalar(lambda w: -scores(k, dk, symmetric(w)), bounds=(0, 1), method='bounded')
    assert best >= -opt.fun-1e-12


def test_symmetric_search_retains_endpoint_adjacent_maximum():
    from remaining_computations import symmetric_search, scores, symmetric
    from symmetric_qfi import stationary_candidates
    channel = ThermometryChannel(10, build_probe_weights(10, 'local_gaussian', 1.))
    k, dk = channel.kernel(.1900882337, 7.3)
    found, w = symmetric_search(k, dk)
    assert 0 < w < 1/64
    expected = max(scores(k, dk, symmetric(stationary_candidates(k, dk))))
    assert found == pytest.approx(expected, rel=1e-10, abs=1e-14)
