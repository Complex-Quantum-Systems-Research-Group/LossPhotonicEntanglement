import numpy as np
from observables import partial_trace_spins, partial_trace_photons
import os
from parameters import SimulationParameters

params = SimulationParameters()

os.makedirs("data_photon", exist_ok=True)

for Temp_spin in temperature_list:
    for delta_t in delta_t_list:
        filename = f"rho_evolved_Nspins={N_spins}_nmax={n_max}_omega1={omega1:1.2f}_omega2={omega2:1.2f}_J{J:1.2f}_delta={delta:1.2f}_T={Temp_spin:1.2f}_tau1={tau_1:1.2f}_tau2={tau_2:1.2f}_delta_t={delta_t:1.2f}_final_t={final_evolution_time:1.2f}.npy"
        filepath_in = os.path.join("data", filename)
        rho = np.load(filepath_in, allow_pickle=True)
        rho_photon = partial_trace_spins(rho,n_max,N_spins)
        filename_out = f"rho_photons_Nspins={N_spins}_nmax={n_max}_omega1={omega1:1.2f}_omega2={omega2:1.2f}_J{J:1.2f}_delta={delta:1.2f}_T={Temp_spin:1.2f}_tau1={tau_1:1.2f}_tau2={tau_2:1.2f}_delta_t={delta_t:1.2f}_final_t={final_evolution_time:1.2f}.npy"
        filepath_out = os.path.join("data_photon", filename_out)
        np.save(filepath_out,rho_photon)
        