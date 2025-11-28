import numpy as np
from observables import partial_trace_spins, partial_trace_photons

# ============================================================================
# PARAMETERS
# ============================================================================
n_max = 4
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
temperature_list = np.arange(0.5,2.0,0.2)
delta_t_list = np.arange(0.5,2.05,0.2)

for Temp_spin in temperature_list:
    for delta_t in delta_t_list:
        filename = f"rho_evolved_Nspins={N_spins}_nmax={n_max}_omega1={omega1:1.2f}_omega2={omega2:1.2f}_J{J:1.2f}_delta={delta:1.2f}_T={Temp_spin:1.2f}_tau1={tau_1:1.2f}_tau2={tau_2:1.2f}_delta_t={delta_t:1.2f}_final_t={final_evolution_time:1.2f}.npy"
        rho = np.load(filename,allow_pickle=True)
        rho_photon = partial_trace_spins(rho,n_max,N_spins)
        filename_out = f"rho_photons_Nspins={N_spins}_nmax={n_max}_omega1={omega1:1.2f}_omega2={omega2:1.2f}_J{J:1.2f}_delta={delta:1.2f}_T={Temp_spin:1.2f}_tau1={tau_1:1.2f}_tau2={tau_2:1.2f}_delta_t={delta_t:1.2f}_final_t={final_evolution_time:1.2f}.npy"
        np.save(filename_out,rho_photon)