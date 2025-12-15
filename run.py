import numpy as np
from hamiltonians import build_total_hamiltonian
from states import bell_photons, thermal_spin_density_matrix
from evolution import evolve_density_matrix, evolve_to_time
from observables import photon_numbers, spin_at_site
from protocol import full_pipeline
import os
from parameters import SimulationParameters
from coherence import OffDiagonalMeasure, RelativeEntropyCoherence

params = SimulationParameters()

os.makedirs("data", exist_ok=True)

for Temp_spin in temperature_list:
    for delta_t in delta_t_list:
        rho_out = full_pipeline(
            n_max, N_spins, omega1, omega2, g1, g2, J, delta,
            Temp_spin, tau_1, delta_t, tau_2, final_evolution_time
        )

        # Save evolved density matrix
        filename = (
            f"rho_evolved_Nspins={N_spins}_nmax={n_max}_omega1={omega1:1.2f}_omega2={omega2:1.2f}"
            f"_J{J:1.2f}_delta={delta:1.2f}_T={Temp_spin:1.2f}_tau1={tau_1:1.2f}"
            f"_tau2={tau_2:1.2f}_delta_t={delta_t:1.2f}_final_t={final_evolution_time:1.2f}.npy"
        )
        filepath = os.path.join("data", filename)
        np.save(filepath, rho_out)

        # --- New measures ---
        offdiag_val = OffDiagonalMeasure.compute(rho_out)
        relentropy_val = RelativeEntropyCoherence.compute(rho_out)

        # Save them with parallel filenames
        np.save(filepath.replace("rho_evolved", "offdiag"), offdiag_val)
        np.save(filepath.replace("rho_evolved", "relentropy"), relentropy_val)