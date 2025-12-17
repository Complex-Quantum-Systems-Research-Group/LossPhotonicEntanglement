from compute_entanglement import von_neumann_entropy, mutual_information
import numpy as np
import os

# ============================================================================
# PARAMETERS
# ============================================================================
n_max = 2
N_spins = 6
omega1 = 1.0
omega2 = 1.0
g1 = 2.0
g2 = 2.0
J = -1.0
delta = 0.5
tau_1 = 0.05
tau_2 = 0.05
final_evolution_time = 2.0

temperature_list = np.arange(0.05, 2.05, 0.05)
delta_t_list = np.arange(0.5, 2.05, 0.05)

# Ensure output directory exists
os.makedirs("data_observable", exist_ok=True)

# Storage for summary
von_neumann_list = []
mutual_information_list = []

# ============================================================================
# MAIN LOOP: compute observables and collect
# ============================================================================
for Temp_spin in temperature_list:
    for delta_t in delta_t_list:
        # Input photon density matrix file
        filename_rho = (
            f"rho_photons_Nspins={N_spins}_nmax={n_max}_omega1={omega1:1.2f}_omega2={omega2:1.2f}"
            f"_J{J:1.2f}_delta={delta:1.2f}_T={Temp_spin:1.2f}_tau1={tau_1:1.2f}"
            f"_tau2={tau_2:1.2f}_delta_t={delta_t:1.2f}_final_t={final_evolution_time:1.2f}.npy"
        )
        filepath_in = os.path.join("data_photon", filename_rho)

        # Load density matrix
        rho_photons = np.load(filepath_in, allow_pickle=True)

        # Compute observables
        von_neumann_val = von_neumann_entropy(rho_photons)
        mutual_info_val = mutual_information(rho_photons, n_max)

        # Save individual results
        filename_von = (
            f"von_neumann_entropy_Nspins={N_spins}_nmax={n_max}_omega1={omega1:1.2f}_omega2={omega2:1.2f}"
            f"_J{J:1.2f}_delta={delta:1.2f}_T={Temp_spin:1.2f}_tau1={tau_1:1.2f}"
            f"_tau2={tau_2:1.2f}_delta_t={delta_t:1.2f}_final_t={final_evolution_time:1.2f}.npy"
        )
        filename_mutual = (
            f"mutual_information_Nspins={N_spins}_nmax={n_max}_omega1={omega1:1.2f}_omega2={omega2:1.2f}"
            f"_J{J:1.2f}_delta={delta:1.2f}_T={Temp_spin:1.2f}_tau1={tau_1:1.2f}"
            f"_tau2={tau_2:1.2f}_delta_t={delta_t:1.2f}_final_t={final_evolution_time:1.2f}.npy"
        )

        filepath_von = os.path.join("data_observable", filename_von)
        filepath_mutual = os.path.join("data_observable", filename_mutual)

        np.save(filepath_von, von_neumann_val)
        np.save(filepath_mutual, mutual_info_val)

        # Collect into summary lists
        von_neumann_list.append((Temp_spin, delta_t, von_neumann_val))
        mutual_information_list.append((Temp_spin, delta_t, mutual_info_val))

# ============================================================================
# SAVE SUMMARY ARRAYS
# ============================================================================
von_neumann_list = np.array(von_neumann_list, dtype=object)
mutual_information_list = np.array(mutual_information_list, dtype=object)

np.save("data_observable/von_neumann_entropy_summary.npy", von_neumann_list)
np.save("data_observable/mutual_information_summary.npy", mutual_information_list)