import numpy as np
import pytest
from scipy.linalg import expm

import config as cfg
from ellipticity_preflight import validate_ellipticity, zero_delay_bell_fidelity
from new_protocol import build_protocol_components, full_pipeline_unitary
from new_evolution import kerr_rotation_unitary
from operators import I2, SY, SZ
from observables import partial_trace_spins
from channel_diagnostics import reconstruct_choi, validate_choi_reconstruction
from validation import assert_rank_two_support


def test_manuscript_four_reference_controls():
    result = validate_ellipticity()
    assert result['ranks'] == [2, 4]
    assert result['zero_delay_error'] < 1e-10


@pytest.mark.parametrize('probe', ['local_gaussian', 'single_site', 'collective'])
@pytest.mark.parametrize('angles,dt', [((.35,.22,.28,.11),1.7),
    ((.4,.25,.4,.25),0.), ((0.,.25,0.,-.15),1.7),
    ((-.4,0.,.3,0.),1.7), ((0.,0.,0.,0.),1.7)])
def test_full_protocol_against_independent_dense_exponentials(probe, angles, dt):
    t1,e1,t2,e2 = angles
    kwargs = dict(n_spins=3, J=1., delta=1., temperature=.7, delta_t=dt,
                  theta1=t1, eta1=e1, theta2=t2, eta2=e2, probe_model=probe)
    c = build_protocol_components(**kwargs)
    M = c['probe_operator']
    U1 = expm(-1j*np.kron(np.kron(t1*SY+e1*SZ,I2),M))
    U2 = expm(-1j*np.kron(np.kron(I2,t2*SY+e2*SZ),M))
    U = U2 @ np.kron(np.eye(4),expm(-1j*dt*c['Hs'])) @ U1
    expected = U @ c['rho_initial'] @ U.conj().T
    np.testing.assert_allclose(full_pipeline_unitary(**kwargs), expected, atol=1e-10, rtol=0)
    np.testing.assert_allclose(c['U1'], U1, atol=1e-10, rtol=0)
    np.testing.assert_allclose(expm(-1j*c['G1']), U1, atol=1e-10, rtol=0)


@pytest.mark.parametrize('angles', [(0,0,0,0),(.4,0,.4,0),(0,.3,0,.3),
                                  (.4,.25,.4,.25),(-.3,.1,.4,-.2)])
def test_public_run_point_zero_delay_and_metadata(monkeypatch, angles):
    from pipeline import run_point
    t1,e1,t2,e2 = angles
    monkeypatch.setattr(cfg, 'eta1', e1)
    monkeypatch.setattr(cfg, 'eta2', e2)
    result = run_point(.7, 0., n_spins=3, theta1=t1, theta2=t2)
    c = build_protocol_components(n_spins=3, J=cfg.J, delta=cfg.delta,
        temperature=.7, delta_t=0., theta1=t1, theta2=t2, eta1=e1, eta2=e2)
    expected = zero_delay_bell_fidelity(t1,e1,t2,e2,c['probe_operator'],c['rho_spin'])
    assert abs(result['bell_fidelity']-expected) < 1e-10
    assert result['eta1'] == e1 and result['eta2'] == e2
    assert abs(sum(result['bell_populations'].values())-1) < 1e-10
    if e1 or e2:
        assert result['p'] is None and result['Im_C'] is None
    else:
        assert_rank_two_support(result['rho_photons'])


def test_elliptical_choi_and_collective_conservation_with_longitudinal_field():
    kw = dict(n_spins=3, J=1., delta=1.2, h_z=.5, temperature=.7,
              theta1=.35, eta1=.22, theta2=.28, eta2=.11)
    result = validate_choi_reconstruction(reconstruct_choi(delta_t=1.7, **kw))
    assert result['cptp_pass']
    assert result['validation']['unitality_residual'] < 1e-10
    a,b = [partial_trace_spins(full_pipeline_unitary(delta_t=dt,
           probe_model='collective', **kw),3) for dt in (0.,1.7)]
    np.testing.assert_allclose(a,b,atol=1e-10,rtol=0)


def test_exchange_rejects_ellipticity():
    with pytest.raises(ValueError, match='ellipticity'):
        full_pipeline_unitary(3,1.,1.,.7,1.7,.4,.4,
                              interaction_type='exchange_benchmark',eta1=.2)


def test_elliptical_reporting_does_not_use_rotation_prediction(tmp_path, monkeypatch, capsys):
    import plot_results
    import analyze_results
    row = dict(theta1=.05, theta2=.05, eta1=.1, eta2=.1, T_kelvin=100., delta_t=1.7)
    monkeypatch.setattr(plot_results, 'compute_D_M', lambda *a, **kw: pytest.fail('invalid theory comparison'))
    plot_results.run_weak_coupling_check([row], tmp_path, tmp_path/'summary.json')
    analyze_results._report_weak_coupling_gate([row])
    assert 'elliptical delay response is exploratory' in capsys.readouterr().out


def test_eta_filename_namespace():
    assert cfg.filename_tag(eta1_value=0) != cfg.filename_tag(eta1_value=.2)


def test_signed_rotation_and_zero_identity():
    M = np.diag([-.7,.2])
    for t in (-.4,0.,.4):
        actual = kerr_rotation_unitary(t,0,M)
        np.testing.assert_allclose(actual, expm(-1j*t*np.kron(np.kron(SY,I2),M)), atol=1e-12)


def test_exported_characteristic_matches_signed_spin_expression(monkeypatch):
    from pipeline import run_point
    from parity_preflight import PHI_PLUS, PSI_MINUS, spin_characteristic_C
    monkeypatch.setattr(cfg, "h_z", .5)
    result = run_point(.76,3.,n_spins=6,eta1=0.,eta2=0.)
    C = spin_characteristic_C(n_spins=6,J=cfg.J,delta=cfg.delta,h_z=.5,
        temperature=.76,delay=3.,theta1=cfg.theta1,theta2=cfg.theta2)
    cross = np.vdot(PHI_PLUS,result['rho_photons'] @ PSI_MINUS)
    assert abs(C.imag) > 1e-6
    assert abs(cross - C.imag/2) < 1e-12
    assert abs(result['Im_C'] - C.imag) < 1e-12
