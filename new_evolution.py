# function that returns unitary based on hamiltonian. change og code

import numpy as np
from scipy.linalg import expm


def unitary_from_hamiltonian(H, t):
    """
    Construct unitary time-evolution operator from a Hamiltonian.

    U(t) = exp(-i H t)

    Parameters:
    -----------
    H : ndarray (dim × dim)
        Hamiltonian
    t : float
        Evolution time

    Returns:
    --------
    U : ndarray (dim × dim)
        Unitary evolution operator
    """
    return expm(-1j * H * t)


def evolve_density_matrix(H, rho0, times):
    """
    Time evolution of density matrix: ρ(t) = U(t) ρ₀ U†(t)
    """
    print(f"Evolving density matrix from t={times[0]:.2f} to t={times[-1]:.2f}")
    print(f"  Number of time steps: {len(times)}")
    print(f"  Hilbert space dimension: {H.shape[0]}")

    dim = H.shape[0]
    n_times = len(times)
    rho_t = np.zeros((n_times, dim, dim), dtype=complex)
    
#change
    for i, t in enumerate(times):
        U = unitary_from_hamiltonian(H, t)
        rho = U @ rho0 @ U.conj().T

        # enforce Hermiticity
        rho = (rho + rho.conj().T) / 2.0

        # enforce trace = 1
        trace = np.trace(rho).real
        if not np.isclose(trace, 1.0, atol=1e-6):
            print(f"  Warning at t={t:.2f}: Tr[ρ] = {trace:.6f}, renormalizing...")
            rho = rho / trace

        rho_t[i] = rho

        if i % max(1, len(times)//10) == 0:
            print(f"  Progress: {100*i/len(times):.0f}%")

    print("  Evolution complete!")
    return rho_t


def evolve_to_time(H, rho0, t):
    """
    Evolve density matrix to a single time point using unitary evolution.
    """
    U = unitary_from_hamiltonian(H, t)
    rho_t = U @ rho0 @ U.conj().T

    # enforce Hermiticity
    rho_t = (rho_t + rho_t.conj().T) / 2.0

    # enforce trace = 1
    trace = np.trace(rho_t).real
    if not np.isclose(trace, 1.0, atol=1e-6):
        print(f"Warning: Tr[ρ] = {trace:.6f} at t={t:.2f}, renormalizing...")
        rho_t = rho_t / trace

    return rho_t
