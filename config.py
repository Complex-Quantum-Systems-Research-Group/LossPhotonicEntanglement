"""Default parameters for the material-anchored EP-MOKS simulation.

Units: hbar = k_B = 1 inside the simulation; J (dimensionless, set to 1.0) is
the internal exchange-energy unit. Real-world time/energy conversion is
provided separately via J_meV and HBAR_MEV_PS -- see delay_fs() below. These
defaults are a reduced one-dimensional proxy for the dominant intrachain
dynamics of a real compound, not a complete microscopic model of it (see
README / manuscript Sec. III.F).
"""
from __future__ import annotations

import numpy as np

# -----------------------------------------------------------------------
# Material anchor: KCuF3, quasi-1D S=1/2 antiferromagnetic Heisenberg chain.
# Intrachain exchange J = 34 meV, near-isotropic (~0.2% x-y anisotropy),
# T_N = 39 K.
# Source: Lake, Tennant, Nagler et al., "Longitudinal Magnetic Dynamics and
# Dimensional Crossover in the Quasi-One-Dimensional, Spin-1/2, Heisenberg
# Antiferromagnet KCuF3," arXiv:cond-mat/0503128.
#
# Only the dominant intrachain XXZ term is modeled. Reported interchain
# exchange, residual (~0.2%) anisotropy, and T_N are NOT encoded as active
# simulation parameters -- doing so would require extending the Hamiltonian
# to include interchain degrees of freedom, which this repository does not
# implement. See manuscript Sec. III.F.
# -----------------------------------------------------------------------
material = "KCuF3"
J_meV = 34.0  # intrachain exchange, arXiv:cond-mat/0503128

# Finite spin chain (dimensionless internal units)
N_spins = 10  # unchanged pending a manuscript update to the finite-size statement
J = 1.0                  # antiferromagnetic sign convention: J>0 = AFM in this
                          # repo's H_s = J*(SxSx+SySy+Delta*SzSz) - h_z*Sz convention
delta = 1.0               # isotropic Heisenberg limit; explicit modeling
                           # approximation -- see manuscript Sec. III.F
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

# -----------------------------------------------------------------------
# Real-unit conversion (reporting only; the sweep itself stays dimensionless).
# hbar in meV*ps (CODATA), tau_J = hbar / J_meV.
# -----------------------------------------------------------------------
HBAR_MEV_PS = 0.6582119569
tau_J_ps = HBAR_MEV_PS / J_meV  # time unit corresponding to dimensionless J=1


def delay_fs(delta_t_dimensionless):
    """Convert a dimensionless delay (in units of hbar/J) to femtoseconds.

    Example: delay_fs(1.0) is the real time corresponding to J*Delta_t/hbar=1
    for the configured material's J_meV.
    """
    return np.asarray(delta_t_dimensionless) * tau_J_ps * 1000.0


def filename_tag() -> str:
    return (
        f"mat={material}_N={N_spins}_J={J:.3g}_Delta={delta:.3g}_hz={h_z:.3g}_"
        f"probe={probe_model}_sigma={probe_sigma_sites:.3g}_"
        f"theta1={theta1:.3g}_theta2={theta2:.3g}_int={interaction_type}"
    )