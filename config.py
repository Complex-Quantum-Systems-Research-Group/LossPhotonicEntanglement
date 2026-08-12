import numpy as np

# System size / Hilbert space
n_max = 2
N_spins = 6

# Hamiltonian parameters
omega1 = 1.0
omega2 = 1.5
g1 = 2.0
g2 = 2.0
J = -1.0
delta = 0.5
interaction_type = "ising_dickie"  # "ising_dickie" or "tavis_cummings"

# Protocol timing
tau_1 = 0.05
tau_2 = 0.05
final_evolution_time = 2.0

# Sweep grids
temperature_list = np.arange(0.05, 2.05, 0.05)
delta_t_list = np.arange(0.5, 2.05, 0.05)


def filename_tag():
    """
    Shared filename fragment so all pipeline stages agree on naming.

    Includes the interaction model and both photon-spin coupling strengths
    so results from different models/couplings cannot silently overwrite
    one another in shared output directories.
    """
    return (
        f"model={interaction_type}_"
        f"Nspins={N_spins}_nmax={n_max}_"
        f"omega1={omega1:1.2f}_omega2={omega2:1.2f}_"
        f"g1={g1:1.2f}_g2={g2:1.2f}_"
        f"J={J:1.2f}_delta={delta:1.2f}_"
        f"tau1={tau_1:1.2f}_tau2={tau_2:1.2f}_"
        f"final_t={final_evolution_time:1.2f}"
    )