"""Default dimensionless parameters for the corrected pre-results simulation.

Units: hbar = k_B = 1 and |J| sets the spin-energy scale.  These defaults are
chosen for numerical proof-of-principle, not fitted to a specific material.
"""
from __future__ import annotations

import numpy as np

# Finite spin chain
N_spins = 6
J = -1.0                 # ferromagnetic exchange; sets energy unit
delta = 1.5              # easy-axis XXZ anisotropy
h_z = 0.0
periodic = False

# Probe/sample coupling
interaction_type = "kerr"          # primary model; "exchange_benchmark" is non-MOKE control
probe_model = "local_gaussian"     # "collective" is an exact conserved control
probe_sigma_sites = 1.0
theta1 = 0.05                       # toy-model effective rotation angle; not material-fitted
theta2 = 0.05                       # same convention as theta1
bell_state = "phi_plus"

# Sweeps.  The results section must not be written until convergence/validation is complete.
temperature_list = np.round(np.arange(0.10, 2.01, 0.10), 10)
delta_t_list = np.round(np.arange(0.00, 4.01, 0.10), 10)

# Output
output_root = "data"


def filename_tag() -> str:
    return (
        f"N={N_spins}_J={J:.3g}_Delta={delta:.3g}_hz={h_z:.3g}_"
        f"probe={probe_model}_sigma={probe_sigma_sites:.3g}_"
        f"theta1={theta1:.3g}_theta2={theta2:.3g}_int={interaction_type}"
    )
