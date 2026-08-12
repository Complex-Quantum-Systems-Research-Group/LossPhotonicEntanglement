#pipeline that doesn't depend on hamiltonian parameters (hamiltonian agnostic); instead take list of 
#unitaries and initial density matrix and return the applciation of these unitaries on the initial density
# matrix. make sure it does same thing as original 
import numpy as np
from hamiltonians import build_total_hamiltonian
from states import bell_photons, thermal_spin_density_matrix
from new_evolution import unitary_from_hamiltonian


def apply_unitaries(rho_initial, unitaries):
    """
    Hamiltonian-agnostic pipeline.
    Applies a sequence of unitaries to an initial density matrix.

    rho -> U_n ... U_2 U_1 rho U_1† U_2† ... U_n†
    """
    rho = rho_initial
    for U in unitaries:
        rho = U @ rho @ U.conj().T
    return rho


def full_pipeline_unitary(
    n_max, N_spins,
    omega1, omega2,
    g1, g2, J, delta,
    interaction_type,
    Temp_spin,
    tau_1, delta_t, tau_2, final_evolution_time
):
    # initial state
    rho_photon = bell_photons(n_max, i=0, j=1)
    rho_matter = thermal_spin_density_matrix(N_spins, J, delta, T=Temp_spin)
    rho_initial = np.kron(rho_photon, rho_matter)

    # Hamiltonians (same structure as original pipeline)
    H1 = build_total_hamiltonian(n_max, N_spins, omega1, omega2, g1, 0, J, delta, interaction_type)
    H2 = build_total_hamiltonian(n_max, N_spins, omega1, omega2, 0,  0, J, delta, interaction_type)
    H3 = build_total_hamiltonian(n_max, N_spins, omega1, omega2, 0, g2, J, delta, interaction_type)
    H4 = build_total_hamiltonian(n_max, N_spins, omega1, omega2, 0,  0, J, delta, interaction_type)

    # Unitaries
    U1 = unitary_from_hamiltonian(H1, tau_1)
    U2 = unitary_from_hamiltonian(H2, delta_t)
    U3 = unitary_from_hamiltonian(H3, tau_2)
    U4 = unitary_from_hamiltonian(H4, final_evolution_time)

    # Hamiltonian-agnostic application
    return apply_unitaries(rho_initial, [U1, U2, U3, U4])