import numpy as np
from scipy.linalg import expm
from operators import (
    create_pauli_matrices,
    create_bosonic_operators,
    operator_at_spin_site,
    operator_at_spin_site_spin_only,
)

def build_bosonic_hamiltonian(n_max, N_spins, omega1, omega2):
    """
    Build H_boson = ω₁ a₁† a₁ + ω₂ a₂† a₂
    
    Returns:
    --------
    H_boson : ndarray
        Bosonic part of Hamiltonian
    """
    print("Building bosonic Hamiltonian...")
    
    a1, a1_dag, n1, id1 = create_bosonic_operators(n_max)
    a2, a2_dag, n2, id2 = create_bosonic_operators(n_max)
    
    dim_spin = 2**N_spins
    
    # H_photon1 = ω₁ n₁ ⊗ I₂ ⊗ I_spins
    H_photon1 = np.kron(n1, np.eye(n_max * dim_spin, dtype=complex))
    
    # H_photon2 = ω₂ I₁ ⊗ n₂ ⊗ I_spins
    H_photon2 = np.kron(np.eye(n_max, dtype=complex), 
                        np.kron(n2, np.eye(dim_spin, dtype=complex)))
    
    H_boson = omega1 * H_photon1 + omega2 * H_photon2
    
    print(f"  Bosonic Hamiltonian shape: {H_boson.shape}")
    return H_boson


def build_spin_hamiltonian_xxz(n_max, N_spins, J, delta):
    """
    Build H_spin = J Σᵢ (σᵢˣσᵢ₊₁ˣ + σᵢʸσᵢ₊₁ʸ + Δ σᵢᶻσᵢ₊₁ᶻ)
    
    NEAREST-NEIGHBOR interactions only (1D chain)
    
    Parameters:
    -----------
    J : float
        Coupling strength
    delta : float
        Anisotropy parameter (Δ=1 is Heisenberg, Δ→∞ is Ising)
        
    Returns:
    --------
    H_spin : ndarray
        Spin-spin interaction Hamiltonian
    """
    print("Building XXZ spin Hamiltonian (nearest-neighbor)...")
    
    sigma_x, sigma_y, sigma_z, _, _, _ = create_pauli_matrices()
    
    dim = n_max * n_max * (2**N_spins)
    H_spin = np.zeros((dim, dim), dtype=complex)
    
    # Sum over nearest-neighbor pairs: (0,1), (1,2), ..., (N-2, N-1)
    for i in range(N_spins - 1):
        j = i + 1  # Next neighbor
        
        # Get operators at sites i and j
        sigma_x_i = operator_at_spin_site(sigma_x, i, N_spins, n_max)
        sigma_x_j = operator_at_spin_site(sigma_x, j, N_spins, n_max)
        
        sigma_y_i = operator_at_spin_site(sigma_y, i, N_spins, n_max)
        sigma_y_j = operator_at_spin_site(sigma_y, j, N_spins, n_max)
        
        sigma_z_i = operator_at_spin_site(sigma_z, i, N_spins, n_max)
        sigma_z_j = operator_at_spin_site(sigma_z, j, N_spins, n_max)
        
        # Add XXZ interaction for this bond
        H_spin += J * (
            sigma_x_i @ sigma_x_j +           # XX term
            sigma_y_i @ sigma_y_j +           # YY term
            delta * (sigma_z_i @ sigma_z_j)   # ΔZZ term
        )
    
    print(f"  Spin Hamiltonian shape: {H_spin.shape}")
    print(f"  Number of bonds: {N_spins - 1}")
    print(f"  J = {J}, Δ = {delta}")
    return H_spin

def build_tavis_cummings_spin_boson_coupling(n_max, N_spins, g1, g2):
    """
    Build Tavis-Cummings coupling:
    H_coupling = g₁(a₁†Σσ₋ + a₁Σσ₊) + g₂(a₂†Σσ₋ + a₂Σσ₊)
    
    Returns:
    --------
    H_coupling : ndarray
        Spin-boson interaction Hamiltonian
    """
    print("Building spin-boson coupling...")
    
    a1, a1_dag, _, _ = create_bosonic_operators(n_max)
    a2, a2_dag, _, _ = create_bosonic_operators(n_max)
    _, _, _, sigma_plus, sigma_minus, _ = create_pauli_matrices()
    
    dim_spin = 2**N_spins
    
    # Build collective spin operators Σσ₊ and Σσ₋
    sigma_plus_total = sum(
        operator_at_spin_site(sigma_plus, i, N_spins, n_max) 
        for i in range(N_spins)
    )
    sigma_minus_total = sum(
        operator_at_spin_site(sigma_minus, i, N_spins, n_max) 
        for i in range(N_spins)
    )
    
    # Extend mode 1 operators to full space
    a1_extended = np.kron(a1, np.eye(n_max * dim_spin, dtype=complex))
    a1dag_extended = np.kron(a1_dag, np.eye(n_max * dim_spin, dtype=complex))
    
    # Extend mode 2 operators to full space
    a2_extended = np.kron(np.eye(n_max, dtype=complex),
                         np.kron(a2, np.eye(dim_spin, dtype=complex)))
    a2dag_extended = np.kron(np.eye(n_max, dtype=complex),
                            np.kron(a2_dag, np.eye(dim_spin, dtype=complex)))
    
    # Build coupling Hamiltonian
    H_coupling = (
        g1 * (a1dag_extended @ sigma_minus_total + a1_extended @ sigma_plus_total) +
        g2 * (a2dag_extended @ sigma_minus_total + a2_extended @ sigma_plus_total)
    )
    
    print(f"  Coupling Hamiltonian shape: {H_coupling.shape}")
    return H_coupling

def build_ising_dickie_spin_boson_coupling(n_max, N_spins, g):
    """
    Build Ising-Dickie coupling:
    H_coupling = g(a₁†Σσᶻ + a₁Σσᶻ) + g(a₂†Σσᶻ + a₂Σσᶻ)
    
    Returns:
    --------
    H_coupling : ndarray
        Spin-boson interaction Hamiltonian
    """
    print("Building Ising-Dickie spin-boson coupling...")
    
    a1, a1_dag, _, _ = create_bosonic_operators(n_max)
    a2, a2_dag, _, _ = create_bosonic_operators(n_max)
    sigma_x, sigma_y, sigma_z, _, _, _ = create_pauli_matrices()
    
    dim_spin = 2**N_spins

    # Build collective spin operators Σσˣ
    sigma_x_total = sum(
        operator_at_spin_site(sigma_x, i, N_spins, n_max) 
        for i in range(N_spins)
    )
    
    # Extend mode 1 operators to full space
    a1_extended = np.kron(a1, np.eye(n_max * dim_spin, dtype=complex))
    a1dag_extended = np.kron(a1_dag, np.eye(n_max * dim_spin, dtype=complex))
    
    # Extend mode 2 operators to full space
    a2_extended = np.kron(np.eye(n_max, dtype=complex),
                         np.kron(a2, np.eye(dim_spin, dtype=complex)))
    a2dag_extended = np.kron(np.eye(n_max, dtype=complex),
                            np.kron(a2_dag, np.eye(dim_spin, dtype=complex)))
    
    # Build coupling Hamiltonian
    H_coupling = (
        g * (a1dag_extended @ sigma_x_total + a1_extended @ sigma_x_total) +
        g * (a2dag_extended @ sigma_x_total + a2_extended @ sigma_x_total)
    )
    
    print(f"  Coupling Hamiltonian shape: {H_coupling.shape}")
    return H_coupling

#
def build_total_hamiltonian(n_max, N_spins, omega1, omega2, g1, g2, 
                           J, delta, interaction_type):
    """
    Build total Hamiltonian by combining all parts
    
    Parameters:
    -----------
    model : str
        'xxz' for XXZ model, 'ising' for pure Ising
        
    Returns:
    --------
    H_total : ndarray
        Complete Hamiltonian
    """
    print(f"\n{'='*60}")
    print(f"{'='*60}")
    print(f"System: {N_spins} spins, {n_max} photons/mode")
    print(f"Hilbert space dimension: {n_max * n_max * (2**N_spins)}")
    print()
    
    # Bosonic part
    H_boson = build_bosonic_hamiltonian(n_max, N_spins, omega1, omega2)
    
    # Spin part
    H_spin = build_spin_hamiltonian_xxz(n_max, N_spins, J, delta)
    
    # Spin-boson coupling

    if interaction_type == 'tavis_cummings':
        H_coupling = build_tavis_cummings_spin_boson_coupling(n_max, N_spins, g1, g2)
    elif interaction_type == 'ising_dickie':
        H_coupling = build_ising_dickie_spin_boson_coupling(n_max, N_spins, g1)

    
    # Total
    H_total = H_boson + H_spin + H_coupling
    
    print()
    print(f"Total Hamiltonian built!")
    print(f"  Shape: {H_total.shape}")
    print(f"  Hermitian: {np.allclose(H_total, H_total.conj().T)}")
    print(f"{'='*60}\n")
    
    return H_total

def build_spin_only_hamiltonian(N_spins, J, delta):
    """
    Build ONLY the spin Hamiltonian (without photon spaces)
    This operates on spin subspace: dimension 2^N × 2^N
    
    Used for computing thermal states
    
    Parameters:
    -----------
    N_spins : int
        Number of spins
    J : float
        Coupling strength
    delta : float
        Anisotropy (for XXZ), ignored for Ising
    model : str
        'xxz' or 'ising'
        
    Returns:
    --------
    H_spin_only : ndarray (2^N × 2^N)
        Spin Hamiltonian in spin subspace only
    """
    
    sigma_x, sigma_y, sigma_z, _, _, _ = create_pauli_matrices()
    
    dim_spin = 2**N_spins
    H_spin_only = np.zeros((dim_spin, dim_spin), dtype=complex)
    
    for i in range(N_spins - 1):
        j = i + 1  # Next neighbor
        
        # Get operators at sites i and j
        sigma_x_i = operator_at_spin_site_spin_only(sigma_x, i, N_spins)
        sigma_x_j = operator_at_spin_site_spin_only(sigma_x, j, N_spins)
        
        sigma_y_i = operator_at_spin_site_spin_only(sigma_y, i, N_spins)
        sigma_y_j = operator_at_spin_site_spin_only(sigma_y, j, N_spins)
        
        sigma_z_i = operator_at_spin_site_spin_only(sigma_z, i, N_spins)
        sigma_z_j = operator_at_spin_site_spin_only(sigma_z, j, N_spins)
        
        # Add XXZ interaction for this bond
        H_spin_only += J * (
            sigma_x_i @ sigma_x_j +           # XX term
            sigma_y_i @ sigma_y_j +           # YY term
            delta * (sigma_z_i @ sigma_z_j)   # ΔZZ term
        )
        
    return H_spin_only



