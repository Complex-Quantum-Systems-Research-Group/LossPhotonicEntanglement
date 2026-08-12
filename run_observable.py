import os
import numpy as np
from entanglement import von_neumann_entropy, mutual_information
import config as cfg

# ============================================================================
# STEP 3: load photon rho, compute entropy + mutual information, save summary
# ============================================================================
os.makedirs("data_observable", exist_ok=True)

von_neumann_list = []
mutual_information_list = []

for Temp_spin in cfg.temperature_list:
    for delta_t in cfg.delta_t_list:
        tag = f"{cfg.filename_tag()}_T={Temp_spin:1.2f}_delta_t={delta_t:1.2f}"

        filepath_in = os.path.join("data_photon", f"rho_photons_{tag}.npy")
        rho_photons = np.load(filepath_in, allow_pickle=True)

        von_neumann_val = von_neumann_entropy(rho_photons)
        mutual_info_val = mutual_information(rho_photons, cfg.n_max)

        np.save(os.path.join("data_observable", f"von_neumann_entropy_{tag}.npy"), von_neumann_val)
        np.save(os.path.join("data_observable", f"mutual_information_{tag}.npy"), mutual_info_val)

        von_neumann_list.append((Temp_spin, delta_t, von_neumann_val))
        mutual_information_list.append((Temp_spin, delta_t, mutual_info_val))

von_neumann_list = np.array(von_neumann_list, dtype=object)
mutual_information_list = np.array(mutual_information_list, dtype=object)

np.save("data_observable/von_neumann_entropy_summary.npy", von_neumann_list)
np.save("data_observable/mutual_information_summary.npy", mutual_information_list)
