import numpy as np

# ============================================================================
# ENTANGLEMENT (formerly entanglement.py)
# ============================================================================

def von_neumann_entropy(rho):
    """Calculate von Neumann entropy S(rho) = -Tr[rho log rho]"""
    eigenvals = np.linalg.eigvalsh(rho)
    eigenvals = eigenvals[eigenvals > 1e-12]
    return -np.sum(eigenvals * np.log(eigenvals))


def partial_trace_mode2(rho_photon, n_max):
    """Trace out photon mode 2 to get mode 1 reduced density matrix."""
    rho_reshaped = rho_photon.reshape(n_max, n_max, n_max, n_max)
    return np.trace(rho_reshaped, axis1=1, axis2=3)


def partial_trace_mode1(rho_photon, n_max):
    """Trace out photon mode 1 to get mode 2 reduced density matrix."""
    rho_reshaped = rho_photon.reshape(n_max, n_max, n_max, n_max)
    return np.trace(rho_reshaped, axis1=0, axis2=2)


def mutual_information(rho_photon, n_max):
    """
    I(1:2) = S(rho_1) + S(rho_2) - S(rho_12)
    Total correlations (classical + quantum) between the two photon modes.
    """
    S_12 = von_neumann_entropy(rho_photon)
    S_1 = von_neumann_entropy(partial_trace_mode2(rho_photon, n_max))
    S_2 = von_neumann_entropy(partial_trace_mode1(rho_photon, n_max))
    return S_1 + S_2 - S_12


# ============================================================================
# COHERENCE (formerly coherence.py)
# ============================================================================

def off_diagonal_measure(rho):
    """Sum of squared magnitudes of off-diagonal entries of rho."""
    off_diag = rho - np.diag(np.diag(rho))
    return np.sum(np.abs(off_diag) ** 2)


def relative_entropy_coherence(rho):
    """C_rel(rho) = S(diag(rho)) - S(rho)"""
    diag_rho = np.diag(np.diag(rho))
    coherence = von_neumann_entropy(diag_rho) - von_neumann_entropy(rho)
    return coherence


def apply_kraus_channel(rho, kraus_ops):
    """Apply a CPTP channel defined by Kraus operators, after checking sum K^dagger K = I."""
    total = sum(k.conj().T @ k for k in kraus_ops)
    if not np.allclose(total, np.eye(total.shape[0])):
        raise ValueError("Kraus operators do not satisfy CPTP condition")
    return sum(k @ rho @ k.conj().T for k in kraus_ops)
