"""
Single entry point for the four sweep stages that were previously separate
scripts (run.py, run_rho_photon.py, run_observable.py, run_coherence.py).
All stages share the same (T, delta_t) loop and config.py parameters.

Usage:
    python pipeline.py evolve       # step 1: sweep + save full rho
    python pipeline.py photon       # step 2: trace out spins -> data_photon/
    python pipeline.py observable   # step 3: von Neumann entropy + mutual info
    python pipeline.py coherence    # step 3 (alt): coherence measures, own sweep
"""
import os
import sys
import numpy as np

import config as cfg
from new_protocol import full_pipeline_unitary
from observables import partial_trace_spins
from measures import (
    von_neumann_entropy,
    mutual_information,
    off_diagonal_measure,
    relative_entropy_coherence,
)


def stage_evolve():
    """Step 1: sweep over (T, delta_t), run full pipeline, save evolved rho."""
    os.makedirs(cfg.interaction_type, exist_ok=True)
    for Temp_spin in cfg.temperature_list:
        for delta_t in cfg.delta_t_list:
            rho_out = full_pipeline_unitary(
                cfg.n_max, cfg.N_spins,
                cfg.omega1, cfg.omega2,
                cfg.g1, cfg.g2, cfg.J, cfg.delta,
                cfg.interaction_type,
                Temp_spin,
                cfg.tau_1, delta_t, cfg.tau_2, cfg.final_evolution_time,
            )
            tag = f"{cfg.filename_tag()}_T={Temp_spin:1.2f}_delta_t={delta_t:1.2f}"
            np.save(os.path.join(cfg.interaction_type, f"rho_evolved_{tag}.npy"), rho_out)


def stage_photon():
    """Step 2: load evolved rho, trace out spins, save photon reduced state."""
    os.makedirs("data_photon", exist_ok=True)
    for Temp_spin in cfg.temperature_list:
        for delta_t in cfg.delta_t_list:
            tag = f"{cfg.filename_tag()}_T={Temp_spin:1.2f}_delta_t={delta_t:1.2f}"
            filepath_in = os.path.join(cfg.interaction_type, f"rho_evolved_{tag}.npy")
            rho = np.load(filepath_in, allow_pickle=True)
            rho_photon = partial_trace_spins(rho, cfg.n_max, cfg.N_spins)
            np.save(os.path.join("data_photon", f"rho_photons_{tag}.npy"), rho_photon)


def stage_observable():
    """Step 3: load photon rho, compute entropy + mutual information, save summary."""
    os.makedirs("data_observable", exist_ok=True)
    von_neumann_list, mutual_information_list = [], []
    for Temp_spin in cfg.temperature_list:
        for delta_t in cfg.delta_t_list:
            tag = f"{cfg.filename_tag()}_T={Temp_spin:1.2f}_delta_t={delta_t:1.2f}"
            rho_photons = np.load(os.path.join("data_photon", f"rho_photons_{tag}.npy"), allow_pickle=True)

            von_neumann_val = von_neumann_entropy(rho_photons)
            mutual_info_val = mutual_information(rho_photons, cfg.n_max)

            np.save(os.path.join("data_observable", f"von_neumann_entropy_{tag}.npy"), von_neumann_val)
            np.save(os.path.join("data_observable", f"mutual_information_{tag}.npy"), mutual_info_val)

            von_neumann_list.append((Temp_spin, delta_t, von_neumann_val))
            mutual_information_list.append((Temp_spin, delta_t, mutual_info_val))

    np.save("data_observable/von_neumann_entropy_summary.npy", np.array(von_neumann_list, dtype=object))
    np.save("data_observable/mutual_information_summary.npy", np.array(mutual_information_list, dtype=object))


def stage_coherence():
    """Step 3 (alt): run full pipeline, compute coherence measures on photon subsystem."""
    os.makedirs("data_coherence", exist_ok=True)
    for Temp_spin in cfg.temperature_list:
        for delta_t in cfg.delta_t_list:
            rho_out = full_pipeline_unitary(
                cfg.n_max, cfg.N_spins,
                cfg.omega1, cfg.omega2,
                cfg.g1, cfg.g2, cfg.J, cfg.delta,
                cfg.interaction_type,
                Temp_spin,
                cfg.tau_1, delta_t, cfg.tau_2, cfg.final_evolution_time,
            )
            rho_photon = partial_trace_spins(rho_out, cfg.n_max, cfg.N_spins)
            offdiag_val = off_diagonal_measure(rho_photon)
            relentropy_val = relative_entropy_coherence(rho_photon)

            tag = f"{cfg.filename_tag()}_T={Temp_spin:1.2f}_delta_t={delta_t:1.2f}"
            np.save(os.path.join("data_coherence", f"rho_evolved_{tag}.npy"), rho_out)
            np.save(os.path.join("data_coherence", f"coherence_measures_{tag}.npy"),
                    {"offdiag": offdiag_val, "relentropy": relentropy_val})


STAGES = {
    "evolve": stage_evolve,
    "photon": stage_photon,
    "observable": stage_observable,
    "coherence": stage_coherence,
}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in STAGES:
        print(f"Usage: python pipeline.py [{'|'.join(STAGES)}]")
        sys.exit(1)
    STAGES[sys.argv[1]]()
