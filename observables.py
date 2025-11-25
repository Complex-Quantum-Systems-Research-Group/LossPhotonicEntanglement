import numpy as np
from scipy.linalg import expm

# ============================================================================
# BASIC BUILDING BLOCKS
# ============================================================================

def create_pauli_matrices():
    """Return Pauli matrices and identity"""
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    sigma_plus = np.array([[0, 1], [0, 0]], dtype=complex)
    sigma_minus = np.array([[0, 0], [1, 0]], dtype=complex)
    id_spin = np.eye(2, dtype=complex)
    
    return sigma_x, sigma_y, sigma_z, sigma_plus, sigma_minus, id_spin


def create_bosonic_operators(n_max):
    """Create annihilation, creation, and number operators for bosonic mode"""
    a = np.diag(np.sqrt(np.arange(1, n_max)), 1)
    a_dag = a.T
    n = a_dag @ a
    id_boson = np.eye(n_max, dtype=complex)
    
    return a, a_dag, n, id_boson


def operator_at_spin_site(op, site, N_spins, n_max):
    """
    Apply operator to specific spin site in full Hilbert space
    Full space: mode1 ⊗ mode2 ⊗ spin1 ⊗ spin2 ⊗ ... ⊗ spinN
    """
    # Start with identity on both photon modes
    result = np.eye(n_max * n_max, dtype=complex)
    
    # Tensor product over all spins
    id_spin = np.eye(2, dtype=complex)
    for i in range(N_spins):
        if i == site:
            result = np.kron(result, op)
        else:
            result = np.kron(result, id_spin)
    
    return result


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
