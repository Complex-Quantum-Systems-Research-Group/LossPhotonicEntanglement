import numpy as np

def von_neumann_entropy(rho):
    """Calculate von Neumann entropy S(ρ) = -Tr[ρ log ρ]"""
    eigenvals = np.linalg.eigvalsh(rho)
    eigenvals = eigenvals[eigenvals > 1e-12]
    S = -np.sum(eigenvals * np.log(eigenvals))
    return S

def partial_trace_mode2(rho_photon, n_max):
    """
    Trace out photon mode 2 to get mode 1 reduced density matrix
    rho_photon has structure: mode1 ⊗ mode2
    Parameters:
    -----------
    rho_photon : ndarray (n_max² × n_max²)
        Two-mode photon density matrix
    n_max : int
        Photon cutoff per mode
    Returns:
    --------
    rho_mode1 : ndarray (n_max × n_max)
        Reduced density matrix for mode 1
    """
    # Reshape: (n_max, n_max, n_max, n_max)
    # Indices: [mode1_bra, mode2_bra, mode1_ket, mode2_ket]
    rho_reshaped = rho_photon.reshape(n_max, n_max, n_max, n_max)
    # Trace over mode 2 (axes 1 and 3)
    rho_mode1 = np.trace(rho_reshaped, axis1=1, axis2=3)
    return rho_mode1

def partial_trace_mode1(rho_photon, n_max):
    """
    Trace out photon mode 1 to get mode 2 reduced density matrix
    Parameters:
    -----------
    rho_photon : ndarray (n_max² × n_max²)
        Two-mode photon density matrix
    n_max : int
        Photon cutoff per mode
    Returns:
    --------
    rho_mode2 : ndarray (n_max × n_max)
        Reduced density matrix for mode 2
    """
    # Reshape: (n_max, n_max, n_max, n_max)
    rho_reshaped = rho_photon.reshape(n_max, n_max, n_max, n_max)
    # Trace over mode 1 (axes 0 and 2)
    rho_mode2 = np.trace(rho_reshaped, axis1=0, axis2=2)
    return rho_mode2

def mutual_information(rho_photon, n_max):
    """
    Calculate mutual information between two photon modes
    I(1:2) = S(ρ₁) + S(ρ₂) - S(ρ₁₂)
    Measures total correlations (classical + quantum) between the two modes
    Parameters:
    -----------
    rho_photon : ndarray (n_max² × n_max²)
        Two-mode photon density matrix (after tracing out spins)
    n_max : int
        Photon cutoff per mode
    Returns:
    --------
    I : float
        Mutual information (I ≥ 0)
        I = 0: modes are independent
        I > 0: modes are correlated
    """
    # Entropy of both modes together
    S_12 = von_neumann_entropy(rho_photon)
    # Entropy of mode 1 alone
    rho_mode1 = partial_trace_mode2(rho_photon, n_max)
    S_1 = von_neumann_entropy(rho_mode1)
    # Entropy of mode 2 alone
    rho_mode2 = partial_trace_mode1(rho_photon, n_max)
    S_2 = von_neumann_entropy(rho_mode2)
    # Mutual information
    I = S_1 + S_2 - S_12
    return I

