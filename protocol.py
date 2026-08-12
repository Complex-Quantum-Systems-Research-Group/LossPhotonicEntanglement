import numpy as np
from hamiltonians import build_total_hamiltonian
from states import bell_photons, thermal_spin_density_matrix
from evolution import evolve_density_matrix, evolve_to_time
from observables import photon_numbers, spin_at_site

def full_pipeline(n_max, N_spins,
                  omega1, omega2,
                  g1, g2,
                  J, delta,
                  interaction_type,
                  Temp_spin,
                  tau_1, delta_t, tau_2,
                  final_evolution_time):

    # --- Initial state ---
    rho_photon = bell_photons(n_max, i=0, j=1)
    rho_matter = thermal_spin_density_matrix(N_spins, J, delta, T=Temp_spin)
    rho_initial = np.kron(rho_photon, rho_matter)

    # --- First interaction: photon 1 couples ---
    H1 = build_total_hamiltonian(
        n_max, N_spins,
        omega1, omega2,
        g1, 0,          # g1 ON, g2 OFF
        J, delta,
        interaction_type
    )
    rho_1 = evolve_to_time(H1, rho_initial, tau_1)

    # --- Bath evolution during Δt (NO photon coupling, but spins evolve!) ---
    H_spin = build_total_hamiltonian(
        n_max, N_spins,
        omega1, omega2,
        0, 0,           # no photon coupling
        J, delta,       # spin Hamiltonian still active
        interaction_type
    )
    rho_2 = evolve_to_time(H_spin, rho_1, delta_t)

    # --- Second interaction: photon 2 couples ---
    H3 = build_total_hamiltonian(
        n_max, N_spins,
        omega1, omega2,
        0, g2,          # g1 OFF, g2 ON
        J, delta,
        interaction_type
    )
    rho_3 = evolve_to_time(H3, rho_2, tau_2)

    # --- Final free evolution (optional) ---
    H_free = build_total_hamiltonian(
        n_max, N_spins,
        omega1, omega2,
        0, 0,
        J, delta,
        interaction_type
    )
    rho_4 = evolve_to_time(H_free, rho_3, final_evolution_time)

    return rho_4