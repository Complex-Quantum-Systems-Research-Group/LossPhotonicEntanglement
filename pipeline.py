"""Validated EP-MOKS parameter sweep.

Run a small validation first:
    python pipeline.py validate

Then, only after the validation and convergence checks pass:
    python pipeline.py sweep
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

import config as cfg
from measures import (
    bell_fidelity,
    concurrence,
    l1_coherence,
    mutual_information,
    purity,
    relative_entropy_coherence,
    von_neumann_entropy,
)
from new_protocol import full_pipeline_unitary
from observables import partial_trace_spins
from states import bell_polarization_state
from validation import assert_density_matrix, density_matrix_diagnostics


def run_point(T: float, delta_t: float) -> dict:
    rho_full = full_pipeline_unitary(
        n_spins=cfg.N_spins,
        J=cfg.J,
        delta=cfg.delta,
        temperature=T,
        delta_t=delta_t,
        theta1=cfg.theta1,
        theta2=cfg.theta2,
        probe_model=cfg.probe_model,
        probe_sigma_sites=cfg.probe_sigma_sites,
        h_z=cfg.h_z,
        periodic=cfg.periodic,
        interaction_type=cfg.interaction_type,
        bell_state=cfg.bell_state,
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


def stage_validate() -> None:
    """Cheap pre-run checks; this is intentionally not the full sweep."""
    test_points = [
        (0.2, 0.0),
        (0.2, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
    ]
    print("Running four-point smoke validation...")
    for T, dt in test_points:
        result = run_point(T, dt)
        print(
            f"T={T:.3g}, dt={dt:.3g}, concurrence={result['concurrence']:.8f}, "
            f"purity={result['purity']:.8f}, MI={result['mutual_information_bits']:.8f}"
        )

    # Exact control: for collective Mz, [H_XXZ,Mz]=0, so the reduced photon
    # state must be independent of the free spin delay (up to roundoff).
    kwargs = dict(
        n_spins=cfg.N_spins,
        J=cfg.J,
        delta=cfg.delta,
        temperature=0.7,
        theta1=cfg.theta1,
        theta2=cfg.theta2,
        probe_model="collective",
        probe_sigma_sites=cfg.probe_sigma_sites,
        h_z=cfg.h_z,
        periodic=cfg.periodic,
        interaction_type="kerr",
        bell_state=cfg.bell_state,
    )
    r0 = partial_trace_spins(full_pipeline_unitary(delta_t=0.0, **kwargs), cfg.N_spins)
    r1 = partial_trace_spins(full_pipeline_unitary(delta_t=1.234, **kwargs), cfg.N_spins)
    err = np.linalg.norm(r0 - r1)
    print(f"collective-Mz exact delay-independence error = {err:.3e}")
    if err > 1e-9:
        raise AssertionError("collective-Mz delay-independence control failed")
    print("Validation passed.")


def stage_sweep() -> None:
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
