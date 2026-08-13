"""Unitary evolution primitives.

Includes three exact algebraic shortcuts that avoid ever forming or
exponentiating/multiplying the full (4 * 2^N)-dimensional joint photon-spin
operator:

- unitary_from_spin_hamiltonian_embedded: since the delay generator is
  I4 (x) Hs, exp(-i(I4 (x) Hs)t) = I4 (x) exp(-i Hs t) exactly, so only the
  spin-only (2^N-dim) Hs ever needs to be exponentiated.

- kerr_rotation_terms / kerr_rotation_unitary: since the 4x4 photon operator
  P = sigma_y^(k) (x) I satisfies P^2 = I4, its eigenvalues are exactly +-1,
  each doubly degenerate, with projectors Pi_+ = (I4+P)/2, Pi_- = (I4-P)/2.
  This gives
    exp(-i theta P (x) M) = Pi_+ (x) exp(-i theta M) + Pi_- (x) exp(+i theta M)
  exactly, so only two exponentials of the spin-only (2^N-dim) magnetization
  operator M are ever needed, never the full generator.

- apply_kron_sum: applies U rho U^dagger for any U expressed as a sum of
  kron(A_i, B_i) terms (A_i on the 4-dim photon-pair space, B_i on the
  2^N-dim spin space) via tensor reshape + einsum, without ever forming or
  multiplying the dense (4*2^N)-dimensional operator. unitary_from_spin_
  hamiltonian_embedded and kerr_rotation_unitary/terms produce exactly this
  kron-sum form.

All three shortcuts were verified to reproduce the generic dense expm /
dense matrix-multiplication path to machine precision (errors ~1e-15-1e-16)
at small system sizes before being wired into new_protocol.py.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from operators import I2, SY, kron_all


def unitary_from_generator(generator, angle):
    """Generic (expensive) path: exp(-i angle * generator) via dense expm.

    Retained for cases without a known closed-form shortcut, e.g. the
    exchange_benchmark interaction, which is a sum of two non-commuting
    tensor-product terms and does not admit the same P^2=I4 reduction as
    the Kerr generator.
    """
    generator = np.asarray(generator, dtype=complex)
    return expm(-1j * float(angle) * generator)


def unitary_from_hamiltonian(H, t):
    return unitary_from_generator(H, t)


def delay_terms(Hs, t, block_dim=4):
    """Kron-sum representation of exp(-i (I_{block_dim} (x) Hs) t): a single
    term (I_{block_dim}, exp(-i Hs t)). Only the spin-only Hs is exponentiated.
    """
    Hs = np.asarray(Hs, dtype=complex)
    small = expm(-1j * float(t) * Hs)
    return [(np.eye(block_dim, dtype=complex), small)]


def unitary_from_spin_hamiltonian_embedded(Hs, t, block_dim=4):
    """Return exp(-i Hs t) embedded as I_{block_dim} (x) exp(-i Hs t), as a
    dense matrix (for validation/inspection; use delay_terms + apply_kron_sum
    for the cheap path that avoids ever forming this dense matrix).
    """
    A, B = delay_terms(Hs, t, block_dim=block_dim)[0]
    return np.kron(A, B)


def kerr_rotation_terms(theta, photon, magnetization):
    """Kron-sum representation of exp(-i theta * sigma_y^(photon) (x) M):
    two terms (Pi_+, exp(-i theta M)) and (Pi_-, exp(+i theta M)), using the
    P^2=I4 projector shortcut (see module docstring). Only two exponentials
    of `magnetization` (spin-only dimension) are ever computed.
    """
    if photon not in (0, 1):
        raise ValueError("photon must be 0 or 1")
    p1 = SY if photon == 0 else I2
    p2 = SY if photon == 1 else I2
    P = kron_all([p1, p2])  # 4x4, P^2 = I4 exactly

    I4 = np.eye(4, dtype=complex)
    Pi_pos = 0.5 * (I4 + P)
    Pi_neg = 0.5 * (I4 - P)

    M = np.asarray(magnetization, dtype=complex)
    U_pos = expm(-1j * float(theta) * M)
    U_neg = expm(+1j * float(theta) * M)

    return [(Pi_pos, U_pos), (Pi_neg, U_neg)]


def kerr_rotation_unitary(theta, photon, magnetization):
    """Return exp(-i theta * sigma_y^(photon) (x) magnetization) as a dense
    matrix (for validation/inspection; use kerr_rotation_terms +
    apply_kron_sum for the cheap path that avoids ever forming this dense
    matrix).
    """
    terms = kerr_rotation_terms(theta, photon, magnetization)
    return sum(np.kron(A, B) for A, B in terms)


def apply_unitary(rho, U):
    """Apply a unitary without artificial renormalization or symmetrization."""
    return U @ rho @ U.conj().T


def apply_unitaries(rho, unitaries):
    """Apply unitaries in the order supplied."""
    out = np.asarray(rho, dtype=complex)
    for U in unitaries:
        out = apply_unitary(out, np.asarray(U, dtype=complex))
    return out


def apply_kron_sum(rho, terms, block_dim=4):
    """Apply U rho U^dagger where U = sum_i kron(A_i, B_i), without ever
    forming or multiplying the dense (block_dim * dim(B))-dimensional
    operator.

    Derivation (rho reshaped as a tensor T[i,a,j,b] with i,j indexing the
    block_dim-dimensional factor and a,b indexing the dim(B)-dimensional
    factor):

      (kron(A,B) @ rho)[i,a,j,b]        = sum_{k,c} A[i,k] B[a,c] T[k,c,j,b]
      (rho @ kron(A,B)^dagger)[i,a,j,b] = sum_{k,c} T[i,a,k,c] conj(A[j,k]) conj(B[b,c])

    each computed as two sequential einsum contractions (one over the
    block_dim index, one over the dim(B) index) rather than a single dense
    (block_dim*dim(B))-dimensional matrix multiplication. Verified against
    the dense apply_unitary path to machine precision (~1e-16) at small
    system sizes before use here.
    """
    dim = terms[0][1].shape[0]
    T = np.asarray(rho, dtype=complex).reshape(block_dim, dim, block_dim, dim)

    L = np.zeros_like(T)
    for A, B in terms:
        temp = np.einsum('ik,kcjb->icjb', A, T, optimize=True)
        L += np.einsum('ac,icjb->iajb', B, temp, optimize=True)

    R = np.zeros_like(T)
    for A, B in terms:
        temp2 = np.einsum('iakc,jk->iajc', L, A.conj(), optimize=True)
        R += np.einsum('iajc,bc->iajb', temp2, B.conj(), optimize=True)

    return R.reshape(block_dim * dim, block_dim * dim)


def apply_kron_sum_sequence(rho, terms_sequence, block_dim=4):
    """Apply a sequence of kron-sum unitaries in order, each via
    apply_kron_sum. `terms_sequence` is an iterable of kron-sum term lists,
    e.g. [kerr_rotation_terms(theta1, 0, M), delay_terms(Hs, dt),
    kerr_rotation_terms(theta2, 1, M)].
    """
    out = np.asarray(rho, dtype=complex)
    for terms in terms_sequence:
        out = apply_kron_sum(out, terms, block_dim=block_dim)
    return out