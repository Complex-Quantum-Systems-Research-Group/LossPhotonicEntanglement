import numpy as np
from hamiltonians import build_total_hamiltonian
from states import bell_photons, thermal_spin_density_matrix
from evolution import evolve_density_matrix, evolve_to_time
from observables import photon_numbers, spin_at_site
def full_pipeline(n_max,N_spins,omega1,omega2,g1,g2,J,delta,Temp_spin,tau_1,delta_t,tau_2,final_evolution_time):
    #first run
    #initial state
    rho_photon = bell_photons(n_max, i=0, j=1)
    rho_matter = thermal_spin_density_matrix(N_spins, J, delta, T=Temp_spin)
    rho_initial = np.kron(rho_photon, rho_matter)
    #Hamiltonian for first run
    g_1 = g1
    g_2 = 0
    H_1 = build_total_hamiltonian(n_max, N_spins, omega1, omega2, g_1, g_2, J, delta)
    #evolve for first run
    rho_1 = evolve_to_time(H_1,rho_initial,tau_1)
    #Hamiltonian for the second run
    g_1 = 0
    g_2 = 0
    H_2 = build_total_hamiltonian(n_max, N_spins, omega1, omega2, g_1, g_2, J, delta)
    #evolve for second run
    rho_2 = evolve_to_time(H_2,rho_1,delta_t)
    #Hamiltonian for third run
    g_1 = 0
    g_2 = g2
    H_3 = build_total_hamiltonian(n_max, N_spins, omega1, omega2, g_1, g_2, J, delta)
    #evolve fo second run
    rho_3 = evolve_to_time(H_3,rho_2,tau_2)
    #Hamiltonian for forth run
    g_1 = 0
    g_2 = 0
    H_4 = build_total_hamiltonian(n_max, N_spins, omega1, omega2, g_1, g_2, J, delta)
    #evolve fo second run
    rho_4 = evolve_to_time(H_4,rho_3,final_evolution_time)
    return rho_4