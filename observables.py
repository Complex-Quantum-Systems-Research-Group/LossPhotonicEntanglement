import numpy as np
from operators import create_pauli_matrices, create_bosonic_operators, operator_at_spin_site

# ============================================================================
# BASIC BUILDING BLOCKS (create_pauli_matrices, create_bosonic_operators,
# operator_at_spin_site now live in operators.py, shared with hamiltonians.py)
# ============================================================================


def expectation_value(rho, operator):
    """
    Calculate expectation value: ⟨O⟩ = Tr[ρ O]
    
    Parameters:
    -----------
    rho : ndarray (dim × dim)
        Density matrix
    operator : ndarray (dim × dim)
        Observable operator
        
    Returns:
    --------
    expectation : float
        Real expectation value
    """
    return np.real(np.trace(rho @ operator))


def photon_number_operators(n_max, N_spins):
    """
    Build photon number operators n₁ and n₂ in full Hilbert space
    
    Returns:
    --------
    n1_op, n2_op : ndarray (dim × dim)
        Number operators for modes 1 and 2
    """
    a1, a1_dag, n1, _ = create_bosonic_operators(n_max)
    a2, a2_dag, n2, _ = create_bosonic_operators(n_max)
    
    dim_spin = 2**N_spins
    
    # n₁ ⊗ I₂ ⊗ I_spins
    n1_op = np.kron(n1, np.eye(n_max * dim_spin, dtype=complex))
    
    # I₁ ⊗ n₂ ⊗ I_spins
    n2_op = np.kron(np.eye(n_max, dtype=complex),
                    np.kron(n2, np.eye(dim_spin, dtype=complex)))
    
    return n1_op, n2_op

# ============================================================================
# BASIC SINGLE-PARTICLE OBSERVABLES
# ============================================================================

def photon_numbers(rho, n_max, N_spins):
    """
    Calculate ⟨n₁⟩ and ⟨n₂⟩
    
    Returns:
    --------
    n1, n2 : float
    """
    n1_op, n2_op = photon_number_operators(n_max, N_spins)
    n1 = expectation_value(rho, n1_op)
    n2 = expectation_value(rho, n2_op)
    return n1, n2

def spin_at_site(rho, site, component, n_max, N_spins):
    """
    Calculate ⟨σᵢᶜ⟩ where c ∈ {x, y, z}
    
    Parameters:
    -----------
    component : str
        'x', 'y', or 'z'
        
    Returns:
    --------
    spin_exp : float
    """
    sigma_x, sigma_y, sigma_z, _, _, _ = create_pauli_matrices()
    
    if component == 'x':
        op = sigma_x
    elif component == 'y':
        op = sigma_y
    elif component == 'z':
        op = sigma_z
    else:
        raise ValueError(f"Unknown component: {component}")
    
    spin_op = operator_at_spin_site(op, site, N_spins, n_max)
    return expectation_value(rho, spin_op)

# ============================================================================
# REDUCED DENSITY MATRICES AND ENTANGLEMENT
# ============================================================================

def partial_trace_spins(rho, n_max, N_spins):
    """
    Trace out spins to get photon reduced density matrix
    
    ρ_photon = Tr_spins[ρ]
    
    Returns:
    --------
    rho_photon : ndarray (n_max² × n_max²)
    """
    dim_photon = n_max * n_max
    dim_spin = 2**N_spins
    
    # Reshape: (dim_photon, dim_spin, dim_photon, dim_spin)
    rho_reshaped = rho.reshape(dim_photon, dim_spin, dim_photon, dim_spin)
    
    # Trace over spin indices (1 and 3)
    rho_photon = np.trace(rho_reshaped, axis1=1, axis2=3)
    
    return rho_photon


def partial_trace_photons(rho, n_max, N_spins):
    """
    Trace out photons to get spin reduced density matrix
    
    ρ_spin = Tr_photons[ρ]
    
    Returns:
    --------
    rho_spin : ndarray (2^N × 2^N)
    """
    dim_photon = n_max * n_max
    dim_spin = 2**N_spins
    
    # Reshape: (dim_photon, dim_spin, dim_photon, dim_spin)
    rho_reshaped = rho.reshape(dim_photon, dim_spin, dim_photon, dim_spin)
    
    # Trace over photon indices (0 and 2)
    rho_spin = np.trace(rho_reshaped, axis1=0, axis2=2)
    
    return rho_spin

# ============================================================================
# ENERGY AND HAMILTONIAN-RELATED
# ============================================================================

def energy_expectation(rho, H):
    """
    Calculate ⟨H⟩ = Tr[ρ H]
    
    Returns:
    --------
    energy : float
    """
    return expectation_value(rho, H)


def energy_variance(rho, H):
    """
    Calculate ΔE² = ⟨H²⟩ - ⟨H⟩²
    
    Returns:
    --------
    var_E : float
    """
    E_avg = expectation_value(rho, H)
    E2_avg = expectation_value(rho, H @ H)
    return E2_avg - E_avg**2