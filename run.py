import numpy as np
from hamiltonians import build_total_hamiltonian
from states import bell_photons, thermal_spin_density_matrix
from new_evolution import evolve_density_matrix, evolve_to_time
from observables import photon_numbers, spin_at_site
from new_protocol import full_pipeline_unitary
import os
# ============================================================================
# PARAMETERS
# ============================================================================
n_max = 2
N_spins = 6
omega1 = 1.0
omega2 = 1.5
g1 = 2.0
g2 = 2.0
J = -1.0
delta = 0.5
interaction_type = "ising_dickie"
tau_1 = 0.05
tau_2 = 0.05 
final_evolution_time = 2.0
temperature_list = np.arange(0.05,2.05,0.05)
delta_t_list = np.arange(0.5,2.05,0.05)

if interaction_type == "ising_dickie":
    name = "ising_dickie"
elif interaction_type == "tavis_cummings":
    name = "tavis_cummings"
    
os.makedirs(name, exist_ok=True)

for Temp_spin in temperature_list:
    for delta_t in delta_t_list:
        rho_out = full_pipeline_unitary(n_max,N_spins,omega1,omega2,g1,g2,J,delta,interaction_type,Temp_spin,tau_1,delta_t,tau_2,final_evolution_time)
        filename = f"rho_evolved_Nspins={N_spins}_nmax={n_max}_omega1={omega1:1.2f}_omega2={omega2:1.2f}_J{J:1.2f}_delta={delta:1.2f}_T={Temp_spin:1.2f}_tau1={tau_1:1.2f}_tau2={tau_2:1.2f}_delta_t={delta_t:1.2f}_final_t={final_evolution_time:1.2f}.npy"
        filepath = os.path.join(name, filename)
        np.save(filepath, rho_out)
