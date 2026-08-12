import os
import numpy as np
from new_protocol import full_pipeline_unitary
from coherence import off_diagonal_measure, relative_entropy_coherence
from observables import partial_trace_spins
import config as cfg

# ============================================================================
# STEP 3 (alternative): run full pipeline, compute coherence measures
# ============================================================================
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
        np.save(os.path.join("data_coherence", f"coherence_measures_{tag}.npy"), {
            "offdiag": offdiag_val,
            "relentropy": relentropy_val,
        })
