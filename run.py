import os
import numpy as np
from new_protocol import full_pipeline_unitary
import config as cfg

# ============================================================================
# STEP 1: sweep over (T, delta_t), run full pipeline, save evolved rho
# ============================================================================
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
        filename = f"rho_evolved_{cfg.filename_tag()}_T={Temp_spin:1.2f}_delta_t={delta_t:1.2f}.npy"
        filepath = os.path.join(cfg.interaction_type, filename)
        np.save(filepath, rho_out)
