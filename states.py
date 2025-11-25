import numpy as np
from scipy.linalg import expm
from hamiltonians import build_spin_only_hamiltonian

def product_photons(n_max, n1_init=0, n2_init=0):
    """
    Create density matrix for product photon state |n1⟩⊗|n2⟩
    
    Returns:
    --------
    rho_photon : ndarray (n_max² × n_max²)
    """
    # Create pure state vector
    psi_mode1 = np.zeros(n_max, dtype=complex)
    psi_mode1[n1_init] = 1.0
    
    psi_mode2 = np.zeros(n_max, dtype=complex)
    psi_mode2[n2_init] = 1.0
    
    psi_photon = np.kron(psi_mode1, psi_mode2)
    
    # ρ = |ψ⟩⟨ψ|
    rho_photon = np.outer(psi_photon, psi_photon.conj())
    
    return rho_photon

product_photons(4, n1_init=1, n2_init=1)

def bell_photons(n_max, i=0, j=1):
    """
    Create density matrix for Bell-like state (|i,i⟩ + |j,j⟩)/√2
    
    Parameters:
    -----------
    n_max : int
        Photon cutoff per mode
    i, j : int
        Fock state numbers (must satisfy 0 ≤ i,j < n_max)
        
    Returns:
    --------
    rho_photon : ndarray (n_max² × n_max²)
    
    Examples:
    ---------
    bell_photons(n_max, 0, 1)  # (|0,0⟩ + |1,1⟩)/√2
    bell_photons(n_max, 0, 2)  # (|0,0⟩ + |2,2⟩)/√2
    bell_photons(n_max, 1, 3)  # (|1,1⟩ + |3,3⟩)/√2
    """
    if i >= n_max or j >= n_max or i < 0 or j < 0:
        raise ValueError(f"i={i} and j={j} must satisfy 0 ≤ i,j < n_max={n_max}")
    
    if i == j:
        raise ValueError(f"i and j must be different (got i=j={i})")
    
    # Create entangled state vector
    psi_photon = np.zeros(n_max * n_max, dtype=complex)
    
    # |i,i⟩ component: index = i * n_max + i
    psi_photon[i * n_max + i] = 1.0 / np.sqrt(2)
    
    # |j,j⟩ component: index = j * n_max + j
    psi_photon[j * n_max + j] = 1.0 / np.sqrt(2)
    
    # ρ = |ψ⟩⟨ψ|
    rho_photon = np.outer(psi_photon, psi_photon.conj())
    
    return rho_photon

def thermal_spin_density_matrix(N_spins, J, delta, T=1.0):
    """
    Compute thermal density matrix: ρ = exp(-β H_spin) / Z
    
    Returns:
    --------
    rho_spin : ndarray (2^N × 2^N)
    """
    # Get spin-only Hamiltonian
    H_spin_only = build_spin_only_hamiltonian(N_spins, J, delta)
    
    # Compute thermal state
    beta = 1.0 / T
    exp_neg_beta_H = expm(-beta * H_spin_only)
    Z = np.trace(exp_neg_beta_H)
    rho_spin = exp_neg_beta_H / Z
    
    return rho_spin


