import numpy as np


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

    Parameters:
    -----------
    op : ndarray
        2x2 spin operator
    site : int
        Which spin (0 to N-1)
    N_spins : int
        Total number of spins
    n_max : int
        Photon cutoff per mode
    """
    result = np.eye(n_max * n_max, dtype=complex)

    id_spin = np.eye(2, dtype=complex)
    for i in range(N_spins):
        if i == site:
            result = np.kron(result, op)
        else:
            result = np.kron(result, id_spin)

    return result


def operator_at_spin_site_spin_only(op, site, N_spins):
    """
    Apply operator to specific spin site in SPIN-ONLY subspace
    (no photon modes involved)

    Spin space: spin1 ⊗ spin2 ⊗ ... ⊗ spinN
    """
    id_spin = np.eye(2, dtype=complex)

    result = op if site == 0 else id_spin

    for i in range(1, N_spins):
        if i == site:
            result = np.kron(result, op)
        else:
            result = np.kron(result, id_spin)

    return result
