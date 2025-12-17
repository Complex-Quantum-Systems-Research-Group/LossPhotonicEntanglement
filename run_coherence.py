import numpy as np
import os

from hamiltonians import build_total_hamiltonian
from states import bell_photons, thermal_spin_density_matrix
from evolution import evolve_density_matrix, evolve_to_time
from observables import photon_numbers, spin_at_site
from protocol import full_pipeline

# Import coherence quantifiers
from coherence import off_diagonal_measure, relative_entropy_coherence, apply_kraus_channel

# ============================================================================
# PARAMETERS
# ============================================================================
n_max = 2
N_spins = 6
omega1 = 1.0
omega2 = 1.0
g1 = 20.0
g2 = 20.0
J = -1.0
delta = 0.5
tau_1 = 0.05
tau_2 = 0.05
final_evolution_time = 2.0

temperature_list = np.arange(0.05, 2.05, 0.05)
delta_t_list = np.arange(0.5, 2.05, 0.05)

os.makedirs("data_coherence", exist_ok=True)

# ============================================================================
# LOOP OVER PARAMETERS
# ============================================================================
for Temp_spin in temperature_list:
    for delta_t in delta_t_list:
        # Run your full pipeline to get evolved density matrix
        rho_out = full_pipeline(
            n_max, N_spins, omega1, omega2, g1, g2, J, delta,
            Temp_spin, tau_1, delta_t, tau_2, final_evolution_time
        )

        # Compute coherence measures
        offdiag_val = off_diagonal_measure(rho_out)
        relentropy_val = relative_entropy_coherence(rho_out)


        # Save density matrix
        filename_rho = (
            f"rho_evolved_Nspins={N_spins}_nmax={n_max}_omega1={omega1:1.2f}_"
            f"omega2={omega2:1.2f}_J{J:1.2f}_delta={delta:1.2f}_T={Temp_spin:1.2f}_"
            f"tau1={tau_1:1.2f}_tau2={tau_2:1.2f}_delta_t={delta_t:1.2f}_"
            f"final_t={final_evolution_time:1.2f}.npy"
        )
        filepath_rho = os.path.join("data_coherence", filename_rho)
        np.save(filepath_rho, rho_out)

        # Save coherence values
        filename_measures = filename_rho.replace("rho_evolved", "coherence_measures")
        filepath_measures = os.path.join("data_coherence", filename_measures)
        np.save(filepath_measures, {"offdiag": offdiag_val, "relentropy": relentropy_val})