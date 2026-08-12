import os
import numpy as np
from observables import partial_trace_spins
import config as cfg

# ============================================================================
# STEP 2: load evolved rho, trace out spins, save photon reduced state
# ============================================================================
os.makedirs("data_photon", exist_ok=True)

for Temp_spin in cfg.temperature_list:
    for delta_t in cfg.delta_t_list:
        tag = f"{cfg.filename_tag()}_T={Temp_spin:1.2f}_delta_t={delta_t:1.2f}"

        filename_in = f"rho_evolved_{tag}.npy"
        filepath_in = os.path.join(cfg.interaction_type, filename_in)
        rho = np.load(filepath_in, allow_pickle=True)

        rho_photon = partial_trace_spins(rho, cfg.n_max, cfg.N_spins)

        filename_out = f"rho_photons_{tag}.npy"
        filepath_out = os.path.join("data_photon", filename_out)
        np.save(filepath_out, rho_photon)
