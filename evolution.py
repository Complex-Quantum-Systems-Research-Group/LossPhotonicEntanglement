import numpy as np
from scipy.linalg import expm
from hamiltonians import build_total_hamiltonian
from states import product_photons,bell_photons,thermal_spin_density_matrix

def evolve_density_matrix(H, rho0, times):
    """
    Time evolution of density matrix: ρ(t) = U(t) ρ₀ U†(t)
    where U(t) = exp(-iHt)
    
    Parameters:
    -----------
    H : ndarray (dim × dim)
        Full Hamiltonian
    rho0 : ndarray (dim × dim)
        Initial density matrix
    times : array
        Time points for evolution
        
    Returns:
    --------
    rho_t : ndarray (n_times, dim, dim)
        Density matrices at each time
    """
    print(f"Evolving density matrix from t={times[0]:.2f} to t={times[-1]:.2f}")
    print(f"  Number of time steps: {len(times)}")
    print(f"  Hilbert space dimension: {H.shape[0]}")
    
    dim = H.shape[0]
    n_times = len(times)
    rho_t = np.zeros((n_times, dim, dim), dtype=complex)
    
    for i, t in enumerate(times):
        # Compute U(t) = exp(-iHt)
        U = expm(-1j * H * t)
        
        # ρ(t) = U(t) ρ₀ U†(t)
        rho = U @ rho0 @ U.conj().T
        
        # Enforce Hermiticity (remove numerical errors)
        rho = (rho + rho.conj().T) / 2.0
        
        # Optionally enforce trace = 1 (should be conserved, but numerical drift)
        trace = np.trace(rho).real
        if not np.isclose(trace, 1.0, atol=1e-6):
            print(f"  Warning at t={t:.2f}: Tr[ρ] = {trace:.6f}, renormalizing...")
            rho = rho / trace
        
        rho_t[i] = rho
        
        # Progress indicator
        if i % max(1, len(times)//10) == 0:
            print(f"  Progress: {100*i/len(times):.0f}%")
    
    print("  Evolution complete!")
    
    return rho_t

