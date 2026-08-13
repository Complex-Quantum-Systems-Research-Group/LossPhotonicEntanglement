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

import json
import os
import sys

import numpy as np

import config as cfg
from hamiltonians import (
    build_spin_hamiltonian_xxz,
    collective_magnetization_z,
    probe_weights_gaussian,
    weighted_magnetization_z,
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
from new_protocol import build_protocol_components, full_pipeline_unitary
from observables import partial_trace_spins
from states import bell_polarization_state, thermal_state_from_hamiltonian
from validation import assert_density_matrix, assert_unitary


_VALIDATION_ATOL = 1e-10


def _assert_close(value: float, target: float, label: str, atol: float = 1e-10) -> None:
    if not np.isclose(value, target, atol=atol, rtol=0.0):
        raise AssertionError(f"{label}: expected {target}, got {value}")


def _protocol_kwargs() -> dict:
    return dict(
        n_spins=cfg.N_spins,
        J=cfg.J,
        delta=cfg.delta,
        probe_sigma_sites=cfg.probe_sigma_sites,
        h_z=cfg.h_z,
        periodic=cfg.periodic,
        interaction_type=cfg.interaction_type,
        bell_state=cfg.bell_state,
    )


def run_point(T: float, delta_t: float) -> dict:
    rho_full = full_pipeline_unitary(
        temperature=T,
        delta_t=delta_t,
        theta1=cfg.theta1,
        theta2=cfg.theta2,
        probe_model=cfg.probe_model,
        **_protocol_kwargs(),
    )
    assert_density_matrix(rho_full)
    rho_p = partial_trace_spins(rho_full, cfg.N_spins)
    assert_density_matrix(rho_p)
    target = bell_polarization_state(cfg.bell_state)
    return {
        "T": float(T),
        "delta_t": float(delta_t),
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
                **_protocol_kwargs(),
            )
            rho_p = partial_trace_spins(rho_full, cfg.N_spins)
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
        cfg.N_spins,
        cfg.J,
        cfg.delta,
        h_z=cfg.h_z,
        periodic=cfg.periodic,
    )

    M_collective = collective_magnetization_z(cfg.N_spins)
    comm_collective = float(np.linalg.norm(Hs @ M_collective - M_collective @ Hs))
    if comm_collective > _VALIDATION_ATOL:
        raise AssertionError(f"[Hs,Mz_collective] != 0: norm={comm_collective}")

    local_weights = probe_weights_gaussian(cfg.N_spins, sigma=cfg.probe_sigma_sites)
    M_local = weighted_magnetization_z(cfg.N_spins, local_weights)
    comm_local = float(np.linalg.norm(Hs @ M_local - M_local @ Hs))
    if comm_local <= 1e-8:
        raise AssertionError(
            "chosen nonuniform Gaussian probe unexpectedly commutes with Hs; "
            f"commutator norm={comm_local}"
        )

    for T in (0.0, float(cfg.temperature_list[0]), float(cfg.temperature_list[-1])):
        assert_density_matrix(thermal_state_from_hamiltonian(Hs, T))

    print(
        "[PASS] commutators/thermal state: "
        f"||[Hs,Mcollective]||={comm_collective:.3e}, "
        f"||[Hs,Mlocal]||={comm_local:.3e}"
    )


def _validate_protocol_unitaries() -> None:
    components = build_protocol_components(
        temperature=float(cfg.temperature_list[len(cfg.temperature_list) // 2]),
        delta_t=float(cfg.delta_t_list[len(cfg.delta_t_list) // 2]),
        theta1=cfg.theta1,
        theta2=cfg.theta2,
        probe_model=cfg.probe_model,
        **_protocol_kwargs(),
    )

    for name in ("U1", "Udelay", "U2"):
        assert_unitary(components[name], atol=_VALIDATION_ATOL)
    print("[PASS] U1, Udelay, and U2 satisfy U^dagger U = I")


def _validate_collective_delay_independence() -> None:
    kwargs = dict(
        n_spins=cfg.N_spins,
        J=cfg.J,
        delta=cfg.delta,
        temperature=float(cfg.temperature_list[len(cfg.temperature_list) // 2]),
        theta1=cfg.theta1,
        theta2=cfg.theta2,
        probe_model="collective",
        probe_sigma_sites=cfg.probe_sigma_sites,
        h_z=cfg.h_z,
        periodic=cfg.periodic,
        interaction_type="kerr",
        bell_state=cfg.bell_state,
    )
    dt_a = 0.0
    dt_b = float(cfg.delta_t_list[-1]) if len(cfg.delta_t_list) else 1.234
    r0 = partial_trace_spins(full_pipeline_unitary(delta_t=dt_a, **kwargs), cfg.N_spins)
    r1 = partial_trace_spins(full_pipeline_unitary(delta_t=dt_b, **kwargs), cfg.N_spins)
    err = float(np.linalg.norm(r0 - r1))
    if err > _VALIDATION_ATOL:
        raise AssertionError(f"collective-Mz delay-independence failed: error={err}")
    print(f"[PASS] collective-Mz delay independence; error={err:.3e}")


def _validate_primary_smoke_points() -> None:
    test_points = [
        (float(cfg.temperature_list[0]), float(cfg.delta_t_list[0])),
        (float(cfg.temperature_list[0]), float(cfg.delta_t_list[-1])),
        (float(cfg.temperature_list[-1]), float(cfg.delta_t_list[0])),
        (float(cfg.temperature_list[-1]), float(cfg.delta_t_list[-1])),
    ]
    for T, dt in test_points:
        result = run_point(T, dt)
        print(
            f"[PASS] smoke T={T:.3g}, dt={dt:.3g}: "
            f"C={result['concurrence']:.8f}, purity={result['purity']:.8f}, "
            f"MI={result['mutual_information_bits']:.8f}"
        )


def stage_validate() -> None:
    """Run the mandatory, cheap preflight controls used to gate the sweep."""
    print("Running mandatory EP-MOKS pre-run validation...")
    _validate_bell_baseline()
    _validate_zero_coupling_identity()
    _validate_commutators_and_thermal_state()
    _validate_protocol_unitaries()
    _validate_collective_delay_independence()
    _validate_primary_smoke_points()
    print("All mandatory pre-run validation checks passed.")


def stage_sweep() -> None:
    # Hard gate: a sweep cannot start unless the same mandatory validation
    # routine advertised by the manuscript succeeds in this process.
    stage_validate()

    os.makedirs(cfg.output_root, exist_ok=True)
    rows = []
    for T in cfg.temperature_list:
        for dt in cfg.delta_t_list:
            result = run_point(float(T), float(dt))
            rho_p = result.pop("rho_photons")
            rows.append(result)
            stem = f"{cfg.filename_tag()}_T={T:.4f}_dt={dt:.4f}"
            np.save(os.path.join(cfg.output_root, f"rho_photons_{stem}.npy"), rho_p)

    np.savez_compressed(
        os.path.join(cfg.output_root, f"summary_{cfg.filename_tag()}.npz"),
        rows=np.array(rows, dtype=object),
    )
    with open(os.path.join(cfg.output_root, f"summary_{cfg.filename_tag()}.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


STAGES = {"validate": stage_validate, "sweep": stage_sweep}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in STAGES:
        print("Usage: python pipeline.py [validate|sweep]")
        raise SystemExit(2)
    STAGES[sys.argv[1]]()
