"""Local estimation of x=log(T/J), with fixed preparation and controls.

All energies are H/J and delays are tJ/hbar. No loss or postselection.
QFI phase reduction applies only to a temperature-independent control basis.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import differential_evolution, minimize

from operators import I2, SX, SY, SZ
from sector_correlations import _sector_operators

BELL = np.array([1, 0, 0, 1], complex) / np.sqrt(2)
CUTOFFS = (1e-15, 1e-13, 1e-11)


class ThermometryChannel:
    """Exact conserved sectors, with no thermal or spectral truncation.

    Cache L at each delay; reuse it for all temperatures and input searches.
    Every magnetization sector is retained, including both spin-flip partners.
    """

    def __init__(self, n, weights, theta1=.4, theta2=.4, eta1=0., eta2=0.,
                 delta=1., h_over_J=0., periodic=False):
        weights = np.asarray(weights, float)
        if n < 2 or weights.shape != (n,) or not np.isfinite(weights).all() or np.abs(weights).sum() == 0:
            raise ValueError("Require N>=2 and finite nonzero probe weights")
        if not np.isfinite([theta1, theta2, eta1, eta2, delta, h_over_J]).all():
            raise ValueError("Controls must be finite")
        self.n = n
        self.weights = weights / np.abs(weights).sum()
        g1, v1 = eigh(theta1 * SY + eta1 * SZ)
        g2, v2 = eigh(theta2 * SY + eta2 * SZ)
        self.basis = np.kron(v1, v2)
        self.bell_control = self.basis.conj().T @ BELL
        self.q_bell = np.abs(self.bell_control)**2
        self.sectors = []
        for k in range(n + 1):
            h, m = _sector_operators(n, k, 1., delta, h_over_J, periodic, self.weights)
            e, v = eigh(h)
            first = [v.T @ (np.exp(-1j * g * m)[:, None] * v) for g in g1]
            second = [np.exp(-1j * g * m) for g in g2]
            self.sectors.append((e, v, first, second))
        self.energies = np.concatenate([s[0] for s in self.sectors])
        self.cache = {}

    def overlaps(self, delay):
        delay = float(delay)
        if not np.isfinite(delay) or delay < 0:
            raise ValueError("Delay must be finite and nonnegative")
        if delay not in self.cache:
            pieces = []
            for e, v, first, second in self.sectors:
                propagated = [v @ (np.exp(-1j * e * delay)[:, None] * a) for a in first]
                branches = np.array([b[:, None] * a for a in propagated for b in second])
                pieces.append(np.einsum('amn,bmn->abn', branches, branches.conj()))
            self.cache[delay] = np.concatenate(pieces, axis=2)
        return self.cache[delay]

    def kernel(self, temperature, delay):
        return thermal_kernel(self.energies, self.overlaps(delay), temperature)


def thermal_populations(energies, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("T/J must be finite and positive for log thermometry")
    e = np.asarray(energies, float)
    # Shift energies also in the derivative, to avoid subtracting a large offset.
    shifted = e - e.min()
    w = np.exp(-shifted / temperature)
    w /= w.sum()
    centered = shifted - w @ shifted
    return w, w * centered / temperature, float(w @ centered**2 / temperature**2)


def thermal_kernel(energies, branch_overlaps, temperature):
    w, dw, _ = thermal_populations(energies, temperature)
    return (np.einsum('abn,n->ab', branch_overlaps, w),
            np.einsum('abn,n->ab', branch_overlaps, dw))


def product_probabilities(u, v):
    return np.array([u*v, u*(1-v), (1-u)*v, (1-u)*(1-v)])


def output_from_probabilities(probabilities, kernel, dkernel):
    q = np.asarray(probabilities, float)
    if q.shape != (4,) or not np.isfinite(q).all() or q.min() < 0 or abs(q.sum()-1) > 1e-10:
        raise ValueError("Input probabilities must belong to the four-outcome simplex")
    a = np.sqrt(q)
    state = np.outer(a, a)
    return kernel * state, dkernel * state


def quantum_fisher_information(rho, drho, cutoff=1e-13):
    if cutoff < 0 or not np.isfinite(cutoff):
        raise ValueError("Invalid QFI cutoff")
    if not np.isfinite(rho).all() or not np.isfinite(drho).all():
        raise ValueError("Nonfinite state or derivative")
    if np.linalg.norm(rho-rho.conj().T) > 1e-10 or np.linalg.norm(drho-drho.conj().T) > 1e-10:
        raise ValueError("Non-Hermitian state or derivative")
    if abs(np.trace(rho)-1) > 1e-10 or abs(np.trace(drho)) > 1e-10:
        raise ValueError("Invalid state or derivative trace")
    e, v = np.linalg.eigh(rho)
    if e.min() < -1e-10:
        raise ValueError("Non-positive output")
    d = v.conj().T @ drho @ v
    denom = e[:, None] + e[None, :]
    mask = denom > cutoff
    return float(2 * np.sum(np.abs(d[mask])**2 / denom[mask]))


def tomography_effects():
    """Nine lab-frame Pauli-pair settings, four outcomes per setting."""
    return np.array([[np.kron((I2+s*a)/2, (I2+t*b)/2)
                      for s in (1, -1) for t in (1, -1)]
                     for a in (SX, SY, SZ) for b in (SX, SY, SZ)])


def classical_fisher_information(rho, drho, effects=None, fractions=None):
    effects = tomography_effects() if effects is None else np.asarray(effects)
    f = np.full(len(effects), 1/len(effects)) if fractions is None else np.asarray(fractions, float)
    if f.shape != (len(effects),) or not np.isfinite(f).all() or f.min() < 0 or abs(f.sum()-1) > 1e-12:
        raise ValueError("Measurement fractions must sum to one incident-pair budget")
    if not np.allclose(effects.sum(axis=1), np.eye(4), atol=1e-12, rtol=0):
        raise ValueError("Each POVM must include all outcomes")
    if not np.allclose(effects, effects.conj().swapaxes(-1, -2)) or np.linalg.eigvalsh(effects).min() < -1e-12:
        raise ValueError("Invalid measurement effects")
    p = np.einsum('soij,ji->so', effects, rho).real
    dp = np.einsum('soij,ji->so', effects, drho).real
    if p.min() < -1e-10 or np.max(np.abs(dp[p <= 0]), initial=0) > 1e-10:
        raise ValueError("Invalid outcome probability or unsupported derivative")
    terms = np.zeros_like(p)
    np.divide(dp**2, p, out=terms, where=p > 0)
    return float(f @ terms.sum(axis=1))


def optimize_inputs(kernel, dkernel, q_bell, seeds=(17, 41, 73), grid_size=21, cutoff=1e-13):
    """Numerical lower bounds only; no certificate of a global optimum."""
    if grid_size < 21 or not seeds:
        raise ValueError("Use at least a 21x21 grid and recorded restart seeds")
    def score(q):
        return quantum_fisher_information(*output_from_probabilities(q, kernel, dkernel), cutoff)
    def batch_score(q):
        # Kernels and final outputs are checked independently. Batch tiny eigh
        # calls for grids and DE populations to avoid Python overhead.
        a = np.sqrt(np.asarray(q))
        inp = a[..., :, None]*a[..., None, :]
        e, v = np.linalg.eigh(kernel*inp)
        d = v.conj().swapaxes(-1, -2) @ (dkernel*inp) @ v
        denom = e[..., :, None]+e[..., None, :]
        terms = np.zeros_like(denom)
        np.divide(2*abs(d)**2, denom, out=terms, where=denom > cutoff)
        return terms.sum(axis=(-1, -2))
    bell = score(q_bell)
    records = []
    def products(size):
        uv = np.array([(u, v) for u in np.linspace(0, 1, size) for v in np.linspace(0, 1, size)])
        values = batch_score(product_probabilities(*uv.T).T)
        candidates = list(zip(values, uv))
        candidates.append((score(product_probabilities(.5, .5)), np.array([.5, .5])))
        return candidates
    candidates = products(grid_size)
    scale = max(bell, max(c[0] for c in candidates), 1e-12)
    objective = lambda uv: -score(product_probabilities(*uv))/scale
    distinct = []
    for _, uv in sorted(candidates, key=lambda c: -c[0]):
        if all(np.linalg.norm(uv-prev) >= .15 for prev in distinct):
            distinct.append(uv)
        if len(distinct) == 6:
            break
    for uv in distinct:
        r = minimize(objective, uv, bounds=[(0, 1)]*2, method='L-BFGS-B', options={'ftol': 1e-12, 'gtol': 1e-8})
        candidates.append((score(product_probabilities(*r.x)), r.x))
        records.append({'method': 'product_local', 'success': bool(r.success), 'message': str(r.message), 'value': candidates[-1][0]})
    for seed in seeds:
        r = differential_evolution(lambda uv: -batch_score(product_probabilities(*uv).T)/scale,
                                   [(0, 1)]*2, seed=int(seed), tol=1e-8, atol=1e-11,
                                   popsize=8, maxiter=150, polish=True, vectorized=True, updating='deferred')
        candidates.append((score(product_probabilities(*r.x)), r.x))
        records.append({'method': 'product_DE', 'seed': int(seed), 'success': bool(r.success), 'message': str(r.message), 'value': candidates[-1][0]})
    best, uv = max(candidates, key=lambda c: c[0])
    refined = bell > best + 1e-10
    if refined:
        extra = products(81)
        for _, start in sorted(extra, key=lambda c: -c[0])[:6]:
            r = minimize(objective, start, bounds=[(0, 1)]*2, method='L-BFGS-B')
            extra.append((score(product_probabilities(*r.x)), r.x))
        best, uv = max(candidates + extra, key=lambda c: c[0])
    q_sep = product_probabilities(*uv)
    # Stick breaking is surjective onto the closed simplex, including faces.
    def simplex(z):
        a, b, c = z
        return np.array([a, (1-a)*b, (1-a)*(1-b)*c, (1-a)*(1-b)*(1-c)])
    all_candidates = [(bell, q_bell), (best, q_sep)] + [(score(q), q) for q in np.eye(4)]
    for seed in seeds:
        r = differential_evolution(lambda z: -batch_score(simplex(z).T)/scale, [(0, 1)]*3,
                                   seed=int(seed), tol=1e-8, atol=1e-11, popsize=8, maxiter=200,
                                   polish=True, vectorized=True, updating='deferred')
        all_candidates.append((score(simplex(r.x)), simplex(r.x)))
        records.append({'method': 'simplex_DE', 'seed': int(seed), 'success': bool(r.success), 'message': str(r.message), 'value': all_candidates[-1][0]})
    all_best, q_all = max(all_candidates, key=lambda c: c[0])
    return {'F_bell': bell, 'F_separable_found': best, 'F_all_inputs_found': all_best,
            'u_opt': float(uv[0]), 'v_opt': float(uv[1]), 'q_all_opt': q_all.tolist(),
            'optimizer_seed': list(seeds), 'optimizer_status': records, 'refined_product_grid': bool(refined)}


def diagnose_input(channel, temperature, delay, vector, cutoff=1e-13):
    """Check a fixed lab-frame preparation; never differentiate an optimizer."""
    a = channel.basis.conj().T @ vector
    inp = np.outer(a, a.conj())
    k, dk = channel.kernel(temperature, delay)
    rho, drho = k*inp, dk*inp
    fq = quantum_fisher_information(rho, drho, cutoff)
    fd_errors, fd_qfi_errors = [], []
    for step in (1e-3, 3e-4, 1e-4):
        plus = channel.kernel(temperature*np.exp(step), delay)[0]
        minus = channel.kernel(temperature*np.exp(-step), delay)[0]
        fd = (plus-minus)*inp/(2*step)
        fd_errors.append(float(np.linalg.norm(fd-drho)))
        fd_qfi_errors.append(abs(quantum_fisher_information(rho, fd, cutoff)-fq))
    sensitivity = max(abs(quantum_fisher_information(rho, drho, c)-fq) for c in CUTOFFS)
    if fd_errors[-1] > 1e-8:
        raise AssertionError("Fixed-input analytic temperature derivative failed")
    lab_rho = channel.basis @ rho @ channel.basis.conj().T
    lab_drho = channel.basis @ drho @ channel.basis.conj().T
    cfi = classical_fisher_information(lab_rho, lab_drho)
    bound = thermal_populations(channel.energies, temperature)[2]
    error = max(sensitivity, fd_qfi_errors[-1], 1e-12)
    if fq > bound + max(error, 1e-9) or cfi > fq + max(error, 1e-9):
        raise AssertionError("Information bound failed")
    if np.linalg.eigvalsh(k).min() < -1e-10 or np.max(abs(np.diag(k)-1)) > 1e-10 or np.max(abs(np.diag(dk))) > 1e-10:
        raise AssertionError("Channel positivity or normalization failed")
    return {'F_Q': fq, 'I_tomography': cfi, 'bath_information_bound': bound,
            'cutoff_sensitivity': sensitivity, 'derivative_absolute_errors': fd_errors,
            'finite_difference_qfi_errors': fd_qfi_errors, 'numerical_error': error,
            'kernel_min_eigenvalue': float(np.linalg.eigvalsh(k).min()),
            'trace_error': float(abs(np.trace(rho)-1)), 'derivative_trace_error': float(abs(np.trace(drho)))}
