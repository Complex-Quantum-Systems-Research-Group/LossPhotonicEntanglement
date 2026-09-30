"""Validated EP-MOKS pre-results workflow.

Commands
--------
python pipeline.py validate
    Run the analytic/limiting-case preflight checks only.

python pipeline.py sweep
    Re-run the same preflight checks first.  The parameter sweep begins only if
    every mandatory check passes.

The validation stage is intentionally cheap relative to the full sweep and is
part of the production execution path, not merely documentation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import hashlib
import subprocess
from pathlib import Path

import numpy as np

import config as cfg
from hamiltonians import (
    build_spin_hamiltonian_xxz,
    collective_magnetization_z,
    probe_weights_gaussian,
    weighted_magnetization_z,
    weighted_magnetization_commutator_xxz,
)
from measures import (
    bell_fidelity,
    concurrence,
    l1_coherence,
    mutual_information,
    purity,
    relative_entropy_coherence,
    von_neumann_entropy,
)
from new_protocol import build_probe_weights, build_protocol_components, full_pipeline_unitary
from observables import partial_trace_spins
from parity_preflight import (
    PHI_PLUS,
    PSI_MINUS,
    analytic_consequences,
    spin_characteristic_C,
)
from states import bell_polarization_state, thermal_state_from_hamiltonian
from validation import assert_density_matrix, assert_unitary, assert_rank_two_support, assert_unitality
from ellipticity_preflight import validate_ellipticity, zero_delay_bell_fidelity
from channel_diagnostics import apply_photon_channel


_VALIDATION_ATOL = 1e-10
_VALIDATION_N = min(cfg.N_spins, 4)


def _assert_close(value: float, target: float, label: str, atol: float = 1e-10) -> None:
    if not np.isclose(value, target, atol=atol, rtol=0.0):
        raise AssertionError(f"{label}: expected {target}, got {value}")


def _protocol_kwargs(n_spins=None) -> dict:
    return dict(
        n_spins=cfg.N_spins if n_spins is None else n_spins,
        J=cfg.J,
        delta=cfg.delta,
        probe_sigma_sites=cfg.probe_sigma_sites,
        h_z=cfg.h_z,
        periodic=cfg.periodic,
        interaction_type=cfg.interaction_type,
        bell_state=cfg.bell_state,
    )


def run_point(T: float, delta_t: float, n_spins=None, probe_model=None,
              probe_sigma_sites=None, theta1=None, theta2=None, eta1=None, eta2=None) -> dict:
    """Compute all diagnostics for one (T, delta_t) protocol point.

    n_spins, probe_model, probe_sigma_sites, theta1, and theta2 default to the current
    config values (None means "use cfg.*"); passing them explicitly lets
    callers (e.g. check_convergence.py) reuse this exact logic -- including
    the density-matrix validation calls -- to probe other N or probe
    profiles without duplicating the computation, so a future change here
    (a bug fix, a new diagnostic) can't silently drift out of sync with a
    parallel reimplementation elsewhere.
    """
    kwargs = _protocol_kwargs()
    n_spins = cfg.N_spins if n_spins is None else n_spins
    kwargs["n_spins"] = n_spins
    if probe_sigma_sites is not None:
        kwargs["probe_sigma_sites"] = probe_sigma_sites
    probe_model = cfg.probe_model if probe_model is None else probe_model
    theta1 = cfg.theta1 if theta1 is None else theta1
    theta2 = cfg.theta2 if theta2 is None else theta2
    eta1 = cfg.eta1 if eta1 is None else eta1
    eta2 = cfg.eta2 if eta2 is None else eta2

    rho_full = full_pipeline_unitary(
        temperature=T,
        delta_t=delta_t,
        theta1=theta1,
        theta2=theta2, eta1=eta1, eta2=eta2,
        probe_model=probe_model,
        **kwargs,
    )
    assert_density_matrix(rho_full)
    rho_p = partial_trace_spins(rho_full, n_spins)
    assert_density_matrix(rho_p)
    target = bell_polarization_state(cfg.bell_state)
    rotation_support = (cfg.interaction_type == "kerr" and cfg.bell_state == "phi_plus"
                        and eta1 == 0.0 and eta2 == 0.0)
    if rotation_support:
        assert_rank_two_support(rho_p)
    bell_vectors = {"phi_plus": [1,0,0,1], "phi_minus": [1,0,0,-1],
                    "psi_plus": [0,1,1,0], "psi_minus": [0,1,-1,0]}
    populations = {name: float(np.vdot(np.asarray(v)/np.sqrt(2),
                   rho_p @ (np.asarray(v)/np.sqrt(2))).real)
                   for name, v in bell_vectors.items()}
    fidelity_zero_delay = None
    if delta_t == 0.0 and cfg.interaction_type == "kerr" and cfg.bell_state == "phi_plus":
        weights = build_probe_weights(n_spins, probe_model, kwargs["probe_sigma_sites"])
        M = weighted_magnetization_z(n_spins, weights)
        H = build_spin_hamiltonian_xxz(n_spins, cfg.J, cfg.delta, cfg.h_z, cfg.periodic)
        fidelity_zero_delay = zero_delay_bell_fidelity(
            theta1, eta1, theta2, eta2, M, thermal_state_from_hamiltonian(H, T))
        _assert_close(bell_fidelity(rho_p, target), fidelity_zero_delay, "zero-delay fidelity")
    return {
        "zero_delay_fidelity_prediction": fidelity_zero_delay,
        "T": float(T),
        "delta_t": float(delta_t),
        "T_kelvin": float(cfg.temperature_kelvin(T)),
        "delta_t_fs": float(cfg.delay_fs(delta_t)),
        "n_spins": n_spins,
        "probe_model": probe_model,
        "probe_sigma_sites": kwargs["probe_sigma_sites"],
        "J": cfg.J,
        "J_meV": cfg.J_meV,
        "delta": cfg.delta,
        "h_z": cfg.h_z,
        "periodic": cfg.periodic,
        "theta1": theta1,
        "theta2": theta2,
        "eta1": eta1,
        "eta2": eta2,
        "theory_status": ("exploratory_ellipticity_delay" if (eta1 or eta2) and delta_t != 0
                          else "exact_zero_delay_control" if (eta1 or eta2)
                          else "rotation_only"),
        "rotation_only_support_applies": rotation_support,
        "bell_mixture_collapse_applies": rotation_support and cfg.h_z == 0.0,
        "bell_populations": populations,
        "p": populations["psi_minus"] if rotation_support else None,
        "Im_C": float(2*np.vdot(PHI_PLUS, rho_p @ PSI_MINUS).real) if rotation_support else None,
        "bell_infidelity": 1.0 - bell_fidelity(rho_p, target),
        "interaction_type": cfg.interaction_type,
        "bell_state": cfg.bell_state,
        "entropy_bits": von_neumann_entropy(rho_p),
        "purity": purity(rho_p),
        "mutual_information_bits": mutual_information(rho_p),
        "concurrence": concurrence(rho_p),
        "l1_coherence": l1_coherence(rho_p),
        "relative_entropy_coherence_bits": relative_entropy_coherence(rho_p),
        "bell_fidelity": bell_fidelity(rho_p, target),
        "rho_photons": rho_p,
    }


def _validate_bell_baseline() -> None:
    rho = bell_polarization_state(cfg.bell_state)
    assert_density_matrix(rho)
    _assert_close(concurrence(rho), 1.0, "Bell concurrence", atol=1e-12)
    _assert_close(mutual_information(rho), 2.0, "Bell mutual information [bits]", atol=1e-12)
    _assert_close(purity(rho), 1.0, "Bell purity", atol=1e-12)
    _assert_close(von_neumann_entropy(rho), 0.0, "Bell pair entropy [bits]", atol=1e-12)
    print("[PASS] Bell baseline: C=1, MI=2 bits, purity=1, S=0")


def _validate_zero_coupling_identity() -> None:
    target = bell_polarization_state(cfg.bell_state)
    temperatures = sorted({
        0.0,
        float(cfg.temperature_list[0]),
        float(cfg.temperature_list[len(cfg.temperature_list) // 2]),
        float(cfg.temperature_list[-1]),
    })
    delays = sorted({
        0.0,
        float(cfg.delta_t_list[0]),
        float(cfg.delta_t_list[len(cfg.delta_t_list) // 2]),
        float(cfg.delta_t_list[-1]),
    })

    worst = 0.0
    for T in temperatures:
        for dt in delays:
            rho_full = full_pipeline_unitary(
                temperature=T,
                delta_t=dt,
                theta1=0.0,
                theta2=0.0,
                probe_model=cfg.probe_model,
                **_protocol_kwargs(_VALIDATION_N),
            )
            rho_p = partial_trace_spins(rho_full, _VALIDATION_N)
            assert_density_matrix(rho_p)
            err = float(np.linalg.norm(rho_p - target))
            worst = max(worst, err)
            if err > _VALIDATION_ATOL:
                raise AssertionError(
                    f"zero-coupling identity failed at T={T}, dt={dt}: ||rho-rho_Bell||={err}"
                )
    print(f"[PASS] zero-coupling identity over {len(temperatures) * len(delays)} controls; worst error={worst:.3e}")


def _validate_commutators_and_thermal_state() -> None:
    Hs = build_spin_hamiltonian_xxz(
        _VALIDATION_N,
        cfg.J,
        cfg.delta,
        h_z=cfg.h_z,
        periodic=cfg.periodic,
    )

    M_collective = collective_magnetization_z(_VALIDATION_N)
    comm_collective = float(np.linalg.norm(Hs @ M_collective - M_collective @ Hs))
    if comm_collective > _VALIDATION_ATOL:
        raise AssertionError(f"[Hs,Mz_collective] != 0: norm={comm_collective}")

    local_weights = probe_weights_gaussian(_VALIDATION_N, sigma=cfg.probe_sigma_sites)
    M_local = weighted_magnetization_z(_VALIDATION_N, local_weights)
    comm_local = float(np.linalg.norm(Hs @ M_local - M_local @ Hs))
    if comm_local <= 1e-8:
        raise AssertionError(
            "chosen nonuniform Gaussian probe unexpectedly commutes with Hs; "
            f"commutator norm={comm_local}"
        )

    comm_formula = weighted_magnetization_commutator_xxz(
        _VALIDATION_N, cfg.J, local_weights, periodic=cfg.periodic,
    )
    formula_residual = float(np.linalg.norm(Hs @ M_local - M_local @ Hs - comm_formula))
    if formula_residual > 1e-12:
        raise AssertionError(
            "weighted-magnetization commutator formula has wrong sign/prefactor: "
            f"residual={formula_residual}"
        )

    for T in (0.0, float(cfg.temperature_list[0]), float(cfg.temperature_list[-1])):
        assert_density_matrix(thermal_state_from_hamiltonian(Hs, T))

    print(
        "[PASS] commutators/thermal state: "
        f"||[Hs,Mcollective]||={comm_collective:.3e}, "
        f"||[Hs,Mlocal]||={comm_local:.3e}, formula residual={formula_residual:.3e}"
    )


def _validate_protocol_unitaries() -> None:
    components = build_protocol_components(
        temperature=float(cfg.temperature_list[len(cfg.temperature_list) // 2]),
        delta_t=float(cfg.delta_t_list[len(cfg.delta_t_list) // 2]),
        theta1=cfg.theta1,
        theta2=cfg.theta2, eta1=cfg.eta1, eta2=cfg.eta2,
        probe_model=cfg.probe_model,
        **_protocol_kwargs(_VALIDATION_N),
    )

    for name in ("U1", "Udelay", "U2"):
        assert_unitary(components[name], atol=_VALIDATION_ATOL)
    print("[PASS] U1, Udelay, and U2 satisfy U^dagger U = I")


def _validate_collective_delay_independence() -> None:
    test_temperature = float(cfg.temperature_list[len(cfg.temperature_list) // 2])
    kwargs = dict(
        n_spins=_VALIDATION_N,
        J=cfg.J,
        delta=cfg.delta,
        temperature=test_temperature,
        theta1=cfg.theta1,
        theta2=cfg.theta2, eta1=cfg.eta1, eta2=cfg.eta2,
        probe_model="collective",
        probe_sigma_sites=cfg.probe_sigma_sites,
        h_z=cfg.h_z,
        periodic=cfg.periodic,
        interaction_type="kerr",
        bell_state=cfg.bell_state,
    )
    dt_a = 0.0
    dt_b = float(cfg.delta_t_list[-1]) if len(cfg.delta_t_list) else 1.234
    Hs = build_spin_hamiltonian_xxz(
        _VALIDATION_N, cfg.J, cfg.delta, h_z=cfg.h_z, periodic=cfg.periodic,
    )
    thermal = thermal_state_from_hamiltonian(Hs, test_temperature)
    magnetization = collective_magnetization_z(_VALIDATION_N)
    mean = np.trace(thermal @ magnetization)
    variance = float(
        np.real(np.trace(thermal @ magnetization @ magnetization) - mean * mean)
    )
    if variance <= 1e-8:
        raise AssertionError(
            "collective delay-independence control is trivial: thermal state "
            f"does not populate multiple magnetization sectors (variance={variance})"
        )
    r0 = partial_trace_spins(full_pipeline_unitary(delta_t=dt_a, **kwargs), _VALIDATION_N)
    r1 = partial_trace_spins(full_pipeline_unitary(delta_t=dt_b, **kwargs), _VALIDATION_N)
    err = float(np.linalg.norm(r0 - r1))
    if err > _VALIDATION_ATOL:
        raise AssertionError(f"collective-Mz delay-independence failed: error={err}")
    print(
        f"[PASS] collective-Mz delay independence; error={err:.3e}, "
        f"Var(Mz_collective)={variance:.3e}"
    )


def _validate_equal_coupling_zero_delay_invariance() -> None:
    """Proposition 4: theta1=theta2=theta, dt=0 => rho_P unchanged, for ANY probe.

    Unlike Proposition 3 (collective probe, conserved M_w, all dt), this holds
    at dt=0 only, but for a generic (including nonuniform) probe operator,
    because the Kerr generator exp(-i theta sigma_y) is real orthogonal so
    (U(mu) x U(mu))|Phi+> = |Phi+> on every eigenspace of M_w.
    """
    target = bell_polarization_state(cfg.bell_state)
    theta = 0.37  # arbitrary nonzero angle; result must hold for any theta
    temperatures = (0.0, float(cfg.temperature_list[0]), float(cfg.temperature_list[-1]))
    probe_models = ("local_gaussian", "single_site", "collective")

    worst = 0.0
    for probe_model in probe_models:
        # Sanity: confirm the probe used is not trivially proportional to the
        # collective one (except when probe_model=="collective" itself), so
        # the check is not vacuously exercising Proposition 3 only.
        weights = build_probe_weights(_VALIDATION_N, probe_model, cfg.probe_sigma_sites)
        if probe_model != "collective" and np.allclose(
            weights, np.full(_VALIDATION_N, 1.0 / _VALIDATION_N), atol=1e-12, rtol=0.0
        ):
            raise AssertionError(f"{probe_model} probe unexpectedly became collective")
        for T in temperatures:
            rho_full = full_pipeline_unitary(
                n_spins=_VALIDATION_N,
                J=cfg.J,
                delta=cfg.delta,
                temperature=T,
                delta_t=0.0,
                theta1=theta,
                theta2=theta,
                probe_model=probe_model,
                probe_sigma_sites=cfg.probe_sigma_sites,
                h_z=cfg.h_z,
                periodic=cfg.periodic,
                interaction_type="kerr",
                bell_state=cfg.bell_state,
            )
            rho_p = partial_trace_spins(rho_full, _VALIDATION_N)
            assert_density_matrix(rho_p)
            err = float(np.linalg.norm(rho_p - target))
            worst = max(worst, err)
            if err > _VALIDATION_ATOL:
                raise AssertionError(
                    "equal-coupling zero-delay invariance failed: "
                    f"probe_model={probe_model}, T={T}, theta={theta}, "
                    f"weights={weights}, ||rho-rho_Bell||={err}"
                )
    print(
        f"[PASS] rotation-only equal-coupling zero-delay invariance over "
        f"{len(probe_models) * len(temperatures)} controls; worst error={worst:.3e}"
    )


def _validate_parity_identity() -> None:
    """Gate Props. 6a/6b and the exact Im(C) consequences before a sweep."""
    n_spins = 6
    theta = 0.4
    delay = 3.0
    temperature = 0.76
    worst = 0.0
    for h_z in (0.0, 0.1, 0.5):
        C = spin_characteristic_C(
            n_spins=n_spins, J=cfg.J, delta=cfg.delta, h_z=h_z,
            temperature=temperature, delay=delay,
            theta1=theta, theta2=theta,
            probe_model=cfg.probe_model,
            probe_sigma_sites=cfg.probe_sigma_sites,
            periodic=cfg.periodic,
        )
        rho_full = full_pipeline_unitary(
            n_spins=n_spins, J=cfg.J, delta=cfg.delta, h_z=h_z,
            temperature=temperature, delta_t=delay,
            theta1=theta, theta2=theta,
            probe_model=cfg.probe_model,
            probe_sigma_sites=cfg.probe_sigma_sites,
            periodic=cfg.periodic, interaction_type="kerr",
            bell_state="phi_plus",
        )
        rho_p = partial_trace_spins(rho_full, n_spins)
        assert_density_matrix(rho_p, atol=1e-12)
        p_a, coh_a, cl1_a = analytic_consequences(C)
        observed = (
            float(np.vdot(PHI_PLUS, rho_p @ PHI_PLUS).real),
            float(abs(np.vdot(PHI_PLUS, rho_p @ PSI_MINUS))),
            l1_coherence(rho_p),
        )
        expected = (1.0 - p_a, coh_a, cl1_a)
        errors = np.abs(np.asarray(observed) - np.asarray(expected))
        worst = max(worst, float(errors.max()))
        if np.any(errors > 1e-12):
            raise AssertionError(
                f"parity analytic identity failed at h_z={h_z}: "
                f"observed={observed}, expected={expected}, errors={errors}"
            )
        rank = int(np.count_nonzero(np.linalg.eigvalsh(rho_p) > 1e-12))
        if rank > 2:
            raise AssertionError(f"rank-two support failed at h_z={h_z}: rank={rank}")
        if h_z == 0.0 and abs(C.imag) > 1e-14:
            raise AssertionError(f"spin-flip parity failed to make C real: Im(C)={C.imag}")
        if h_z != 0.0 and abs(C.imag) <= 1e-6:
            raise AssertionError(f"broken-parity control is insensitive at h_z={h_z}: Im(C)={C.imag}")
    print(
        "[PASS] rotation-only parity identity/rank-two support at h_z=0,0.1,0.5; "
        f"worst analytic residual={worst:.3e}"
    )


def _validate_primary_smoke_points() -> None:
    """Exercise configured parameters through the production algorithm at small N."""
    test_points = [
        (float(cfg.temperature_list[0]), float(cfg.delta_t_list[0])),
        (float(cfg.temperature_list[0]), float(cfg.delta_t_list[-1])),
        (float(cfg.temperature_list[-1]), float(cfg.delta_t_list[0])),
        (float(cfg.temperature_list[-1]), float(cfg.delta_t_list[-1])),
    ]
    for T, dt in test_points:
        result = run_point(T, dt, n_spins=_VALIDATION_N)
        print(
            f"[PASS] smoke N={_VALIDATION_N}, T={T:.3g}, dt={dt:.3g}: "
            f"C={result['concurrence']:.8f}, purity={result['purity']:.8f}, "
            f"MI={result['mutual_information_bits']:.8f}"
        )


def _validate_production_smoke_point() -> None:
    """Run one configured N point through the actual production-size path."""
    T = float(cfg.temperature_list[0])
    dt = float(cfg.delta_t_list[-1])
    result = run_point(T, dt, n_spins=cfg.N_spins)
    print(
        f"[PASS] production smoke N={cfg.N_spins}, T={T:.3g}, dt={dt:.3g}: "
        f"C={result['concurrence']:.8f}, purity={result['purity']:.8f}, "
        f"MI={result['mutual_information_bits']:.8f}"
    )


def stage_validate() -> dict:
    """Run analytic controls plus one production-size point to gate the sweep."""
    print(
        "Running mandatory EP-MOKS pre-run validation "
        f"(analytic controls N={_VALIDATION_N}; production smoke N={cfg.N_spins})..."
    )
    _validate_bell_baseline()
    _validate_zero_coupling_identity()
    _validate_commutators_and_thermal_state()
    _validate_protocol_unitaries()
    _validate_collective_delay_independence()
    _validate_equal_coupling_zero_delay_invariance()
    _validate_parity_identity()
    elliptical_checks = validate_ellipticity()
    print(f"[PASS] elliptical reference controls: {elliptical_checks}")
    protocol = dict(n_spins=_VALIDATION_N, J=cfg.J, delta=cfg.delta,
                    temperature=0.7, delta_t=1.7, theta1=cfg.theta1, theta2=cfg.theta2,
                    eta1=cfg.eta1, eta2=cfg.eta2, probe_model=cfg.probe_model,
                    probe_sigma_sites=cfg.probe_sigma_sites, h_z=cfg.h_z, periodic=cfg.periodic)
    residual = assert_unitality(apply_photon_channel(np.eye(4), **protocol))
    print(f"[PASS] configured Kerr channel unitality residual={residual:.3e}")
    _validate_primary_smoke_points()
    _validate_production_smoke_point()
    print("All mandatory pre-run validation checks passed.")
    return {"passed": True, "tolerance": _VALIDATION_ATOL,
            "ellipticity_reference": elliptical_checks,
            "configured_kerr_unitality_residual": residual}


def _run_sweep_campaign(temperatures, delays, theta1, theta2, campaign, validation=None) -> None:
    """Run and persist one explicitly tagged parameter campaign."""
    tag = cfg.filename_tag(theta1, theta2)
    rows = []
    for T in temperatures:
        for dt in delays:
            result = run_point(
                float(T), float(dt), theta1=float(theta1), theta2=float(theta2)
            )
            rho_p = result.pop("rho_photons")
            result["campaign"] = campaign
            rows.append(result)
            campaign_suffix = "" if campaign == "production" else f"_campaign={campaign}"
            stem = f"{tag}{campaign_suffix}_T={T:.4f}_dt={dt:.4f}"
            np.save(os.path.join(cfg.output_root, f"rho_photons_{stem}.npy"), rho_p)

    summary_stem = (
        f"summary_{tag}" if campaign == "production"
        else f"summary_{tag}_campaign={campaign}"
    )
    sources = sorted(Path(__file__).parent.glob("*.py"))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    manifest = {
        "code_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "config_source": Path(cfg.__file__).read_text(encoding="utf-8"),
        "parameters": {k: v for k, v in rows[0].items() if k in (
            "n_spins", "J", "J_meV", "delta", "h_z", "periodic", "probe_model",
            "probe_sigma_sites", "theta1", "theta2", "eta1", "eta2", "bell_state", "interaction_type")},
        "probe_weights": build_probe_weights(cfg.N_spins, cfg.probe_model, cfg.probe_sigma_sites).tolist(),
        "units": "hbar=k_B=1; energies in J; delays in hbar/J; M=sum(w_i sigma_z_i)",
        "validation": validation,
        "theory_status": sorted({row["theory_status"] for row in rows}),
        "temperatures": [float(t) for t in temperatures], "delays": [float(t) for t in delays],
    }
    with open(os.path.join(cfg.output_root, f"manifest_{tag}_campaign={campaign}.json"),
              "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    np.savez_compressed(
        os.path.join(cfg.output_root, f"{summary_stem}.npz"),
        rows=np.array(rows, dtype=object),
    )
    with open(
        os.path.join(cfg.output_root, f"{summary_stem}.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(rows, handle, indent=2)
    print(
        f"Saved {len(rows)} points for campaign={campaign}, "
        f"theta1={theta1:.3g}, theta2={theta2:.3g}"
    )


def _sample_grid(values, count, label):
    """Select ``count`` evenly spaced existing grid values, including endpoints."""
    if count is None:
        return values
    if not 1 <= count <= len(values):
        raise ValueError(f"--n-{label} must be between 1 and {len(values)}")
    indices = np.linspace(0, len(values) - 1, count).round().astype(int)
    return np.asarray(values)[np.unique(indices)]


def stage_sweep(subgrid=False, campaign=None, n_temps=None, n_delays=None) -> None:
    # Hard gate: a sweep cannot start unless the same mandatory validation
    # routine advertised by the manuscript succeeds in this process.
    if subgrid and (cfg.eta1 != 0.0 or cfg.eta2 != 0.0):
        raise ValueError("weak subgrid uses the rotation-only prediction; set eta1=eta2=0")
    validation = stage_validate()

    os.makedirs(cfg.output_root, exist_ok=True)
    if subgrid:
        temperatures = [cfg.temperature_list[0], cfg.temperature_list[-1]]
        indices = np.linspace(0, len(cfg.delta_t_list) - 1, 10).round().astype(int)
        delays = cfg.delta_t_list[np.unique(indices)]
        print(
            "Running weak-coupling subgrid: "
            f"{len(temperatures)} temperatures x {len(delays)} delays x "
            f"{len(cfg.weak_coupling_theta_values)} theta values"
        )
        for theta in cfg.weak_coupling_theta_values:
            _run_sweep_campaign(
                temperatures, delays, theta, theta,
                campaign="weak_subgrid", validation=validation,
            )
        return

    if campaign is not None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", campaign):
            raise ValueError("campaign must contain only letters, digits, '.', '_' or '-'")
        temperatures = _sample_grid(cfg.temperature_list, n_temps, "temps")
        delays = _sample_grid(cfg.delta_t_list, n_delays, "delays")
        print(
            f"Running campaign={campaign}: {len(temperatures)} temperatures x "
            f"{len(delays)} delays"
        )
        _run_sweep_campaign(
            temperatures, delays, cfg.theta1, cfg.theta2, campaign=campaign, validation=validation,
        )
        return

    _run_sweep_campaign(
        cfg.temperature_list, cfg.delta_t_list, cfg.theta1, cfg.theta2,
        campaign="production", validation=validation,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate", "sweep"))
    parser.add_argument(
        "--subgrid", action="store_true",
        help="run the coarse three-theta perturbative campaign (sweep only)",
    )
    parser.add_argument("--campaign", help="tag a custom sampled sweep")
    parser.add_argument("--n-temps", type=int, help="evenly sample this many configured temperatures")
    parser.add_argument("--n-delays", type=int, help="evenly sample this many configured delays")
    parser.add_argument("--eta1", type=float, default=cfg.eta1, help="photon 1 ellipticity angle")
    parser.add_argument("--eta2", type=float, default=cfg.eta2, help="photon 2 ellipticity angle")
    parser.add_argument("--n-spins", type=int, default=cfg.N_spins)
    args = parser.parse_args()
    if not np.isfinite([args.eta1, args.eta2]).all():
        parser.error("ellipticity angles must be finite")
    if args.n_spins < 2:
        parser.error("--n-spins must be >= 2")
    cfg.eta1, cfg.eta2, cfg.N_spins = args.eta1, args.eta2, args.n_spins
    _VALIDATION_N = min(cfg.N_spins, 4)
    if args.subgrid and args.stage != "sweep":
        parser.error("--subgrid is only valid with the sweep stage")
    if args.subgrid and args.campaign:
        parser.error("--subgrid and --campaign are mutually exclusive")
    if (args.n_temps is not None or args.n_delays is not None) and not args.campaign:
        parser.error("--n-temps/--n-delays require --campaign to protect production output")
    if args.campaign and args.stage != "sweep":
        parser.error("--campaign is only valid with the sweep stage")
    if args.stage == "validate":
        stage_validate()
    else:
        stage_sweep(
            subgrid=args.subgrid, campaign=args.campaign,
            n_temps=args.n_temps, n_delays=args.n_delays,
        )
