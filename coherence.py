import numpy as np

def off_diagonal_measure(rho):
    """
    Compute the sum of squares of off-diagonal matrix elements.
    
    Parameters
    ----------
    rho : ndarray (d × d)
        Density matrix
    
    Returns
    -------
    measure : float
        Sum of squared magnitudes of off-diagonal entries
    """
    off_diag = rho - np.diag(np.diag(rho))
    measure = np.sum(np.abs(off_diag)**2)
    return measure


def relative_entropy_coherence(rho):
    """
    Compute the relative entropy of coherence:
    C_rel(ρ) = S(diag(ρ)) - S(ρ)
    
    Parameters
    ----------
    rho : ndarray (d × d)
        Density matrix
    
    Returns
    -------
    coherence : float
        Relative entropy of coherence
    """
    diag_rho = np.diag(np.diag(rho))

    def von_neumann_entropy(matrix):
        eigvals = np.linalg.eigvalsh(matrix)
        eigvals = eigvals[eigvals > 1e-12]  # filter tiny negatives
        return -np.sum(eigvals * np.log(eigvals))

    coherence = von_neumann_entropy(diag_rho) - von_neumann_entropy(rho)
    return coherence


def apply_kraus_channel(rho, kraus_ops):
    """
    Apply a CPTP channel defined by Kraus operators.
    
    Parameters
    ----------
    rho : ndarray (d × d)
        Input density matrix
    kraus_ops : list of ndarray
        List of Kraus operators (each d × d)
    
    Returns
    -------
    rho_out : ndarray (d × d)
        Output density matrix after channel
    """
    # Check CPTP condition: sum K†K = I
    total = sum([k.conj().T @ k for k in kraus_ops])
    if not np.allclose(total, np.eye(total.shape[0])):
        raise ValueError("Kraus operators do not satisfy CPTP condition")

    rho_out = sum([k @ rho @ k.conj().T for k in kraus_ops])
    return rho_out