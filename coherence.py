# measures.py
import numpy as np

class OffDiagonalMeasure:
    """
    Compute the sum of squares of off-diagonal matrix elements.
    """
    @staticmethod
    def compute(rho: np.ndarray) -> float:
        # Zero out diagonal
        off_diag = rho - np.diag(np.diag(rho))
        return np.sum(np.abs(off_diag)**2)


class RelativeEntropyCoherence:
    """
    Compute the relative entropy of coherence:
    S(diag(rho)) - S(rho)
    where S is von Neumann entropy.
    """
    @staticmethod
    def compute(rho: np.ndarray) -> float:
        # Diagonal density matrix
        diag_rho = np.diag(np.diag(rho))

        def von_neumann_entropy(matrix):
            eigvals = np.linalg.eigvalsh(matrix)
            eigvals = eigvals[eigvals > 1e-12]  # filter tiny negatives
            return -np.sum(eigvals * np.log(eigvals))

        return von_neumann_entropy(diag_rho) - von_neumann_entropy(rho)