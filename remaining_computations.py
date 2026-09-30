"""Reproducible local follow-up: cached midpoint, field escape, rational fits.

Run with the project's Python. Outputs are checkpointed after each field.
Field reweighting is exact: -h Sz is constant in each conserved sector, so
its phase cancels in branch overlaps. Fresh N=10 channels verify this reuse.
"""
from pathlib import Path
from math import comb
import hashlib
import json
import platform
import sys
import time

import numpy as np
from scipy.optimize import differential_evolution, minimize_scalar

from thermometry import ThermometryChannel, thermal_kernel

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reports' / 'remaining_computations'
FIELDS = [0., .01, .03, .1, .3, .5]
PERM = [3, 2, 1, 0]


def scores(k, dk, q, cutoff=1e-13):
    a = np.sqrt(np.asarray(q))
    p = a[..., :, None] * a[..., None, :]
    e, v = np.linalg.eigh(k*p)
    d = v.conj().swapaxes(-1, -2) @ (dk*p) @ v
    denom = e[..., :, None] + e[..., None, :]
    terms = np.zeros_like(denom)
    np.divide(2*abs(d)**2, denom, out=terms, where=denom > cutoff)
    return terms.sum(axis=(-1, -2))


def symmetric(w):
    w = np.asarray(w)
    return np.array([(1-w)/2, w/2, w/2, (1-w)/2]).T


def simplex(z):
    a, b, c = z
    return np.array([a, (1-a)*b, (1-a)*(1-b)*c, (1-a)*(1-b)*(1-c)]).T


def symmetric_search(k, dk):
    ws = np.linspace(0, 1, 65)
    fs = scores(k, dk, symmetric(ws))
    candidates = [(float(f), float(w)) for f, w in zip(fs, ws)]
    scale = max(float(fs.max()), 1e-12)
    for i in range(len(ws)):
        if (i == 0 or fs[i] >= fs[i-1]) and (i == len(ws)-1 or fs[i] >= fs[i+1]):
            r = minimize_scalar(lambda w: -scores(k, dk, symmetric(w))/scale,
                                bounds=(ws[max(0, i-1)], ws[min(len(ws)-1, i+1)]), method='bounded',
                                options={'xatol': 1e-12})
            candidates.append((float(-r.fun*scale), float(r.x)))
    return max(candidates)


def search(k, dk):
    fw, w = symmetric_search(k, dk)
    scale = max(fw, 1e-12)
    best, q = fw, symmetric(w)
    records = []
    if fw > 1e-12:
        for seed in (17, 41, 73):
            r = differential_evolution(lambda z: -scores(k, dk, simplex(z))/scale,
                [(0, 1)]*3, seed=seed, tol=1e-9, atol=1e-12, popsize=8,
                maxiter=200, polish=True, vectorized=True, updating='deferred')
            value = float(scores(k, dk, simplex(r.x)))
            records.append({'seed': seed, 'value': value, 'success': bool(r.success),
                            'iterations': int(r.nit)})
            if value > best:
                best, q = value, simplex(r.x)
    cutoff_error = max(abs(float(scores(k, dk, q, c))-best) for c in (1e-15, 1e-11))
    return dict(F_symmetric=fw, w_star=w, F_full=best, q_star=q.tolist(),
                escape=1-fw/best if best > 1e-12 else None,
                asymmetry=float(np.abs(q-q[PERM]).sum()),
                cutoff_error=cutoff_error, restarts=records)


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def rational_fit(k, dk):
    # Fit on one grid; assess on interlaced held-out points, never training error.
    train = np.linspace(.01, .99, 41)
    test = np.linspace(.015, .985, 80)
    y = scores(k, dk, symmetric(train))
    scale = float(y.max())
    if scale <= 1e-12:
        return None
    y = y/scale
    truth = scores(k, dk, symmetric(test))/scale
    results = {}
    for degree in (2, 3, 4):
        v = np.polynomial.polynomial.polyvander(train, degree)
        mat = np.column_stack([v, -y[:, None]*v[:, 1:]])
        coef = np.linalg.lstsq(mat, y, rcond=None)[0]
        num, den = coef[:degree+1], np.r_[1., coef[degree+1:]]
        pred = np.polynomial.polynomial.polyval(test, num)/np.polynomial.polynomial.polyval(test, den)
        results[str(degree)] = float(np.max(abs(pred-truth)))
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    manifest = dict(status='running', executable=sys.executable, host=platform.node(),
                    cwd=str(ROOT), fields=FIELDS, seeds=[17, 41, 73],
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    optimization='Three seeded full three-parameter simplex DE searches; symmetric candidate retained; numerical lower bounds only')
    dump(OUT/'manifest.json', manifest)
    for eta in (0., .25):
        folder = ROOT/'reports'/'local_thermometry'/f'N10_eta{eta:g}'
        cfg = json.loads((folder/'configuration.json').read_text())
        cache = np.load(folder/'kernels.npz')
        assert cfg['N'] == 10 and cfg['h_over_J'] == 0 and cfg['delta'] == 1
        assert cfg['controls'] == dict(theta1=.4, theta2=.4, eta1=eta, eta2=eta)
        energies, overlaps, delays = [cache[s] for s in ('energies', 'branch_overlaps', 'delays')]
        magnetizations = np.concatenate([np.full(comb(10, k), (10-2*k)/2) for k in range(11)])
        checks = []
        for field in (0., .5):
            direct = ThermometryChannel(10, cfg['weights'], eta1=eta, eta2=eta, h_over_J=field)
            for index in (0, 17, 120):
                a = thermal_kernel(energies-field*magnetizations, overlaps[index], .7)
                b = direct.kernel(.7, float(delays[index]))
                residual = max(float(np.linalg.norm(x-y)) for x, y in zip(a, b))
                assert residual < 1e-12, residual
                if field == 0:
                    assert max(np.linalg.norm(x-x[np.ix_(PERM, PERM)]) for x in b) < 1e-13
                checks.append(dict(field=field, delay=float(delays[index]), residual=residual))
        dump(OUT/f'cache_validation_eta{eta:g}.json', checks)
        midpoint, fits = [], []
        for temperature in cfg['temperatures']:
            for delay, overlap in zip(delays, overlaps):
                k, dk = thermal_kernel(energies, overlap, temperature)
                qbell = np.array([eta**2, .4**2, .4**2, eta**2])/(2*(eta**2+.4**2))
                fmid, fbell = scores(k, dk, [symmetric(.5), qbell])
                midpoint.append(dict(T=temperature, delay=float(delay), F_mid=float(fmid), F_bell=float(fbell),
                    ratio=float(fmid/fbell) if fbell > 1e-12 else None,
                    difference=float(fmid-fbell)))
                fit = rational_fit(k, dk)
                if fit is not None:
                    fits.append(dict(T=temperature, delay=float(delay), errors=fit))
        dump(OUT/f'midpoint_eta{eta:g}.json', midpoint)
        dump(OUT/f'rational_fits_eta{eta:g}.json', fits)
        print(f'eta={eta:g}: midpoint and rational fits complete', flush=True)
        for field in FIELDS:
            path = OUT/f'escape_eta{eta:g}_h{field:g}.json'
            if path.exists():
                print(f'Reusing {path.name}', flush=True)
                continue
            rows = []
            for temperature in cfg['temperatures']:
                for delay, overlap in zip(delays, overlaps):
                    k, dk = thermal_kernel(energies-field*magnetizations, overlap, temperature)
                    row = dict(T=temperature, delay=float(delay), eta=eta, field=field)
                    for name, matrix in [('r_K', k), ('r_dK', dk)]:
                        norm = np.linalg.norm(matrix)
                        row[name] = float(np.linalg.norm(matrix-matrix[np.ix_(PERM, PERM)])/norm) if norm > 1e-14 else None
                    row.update(search(k, dk))
                    rows.append(row)
                print(f'eta={eta:g} h={field:g} T={temperature:g}: {len(rows)}/968; elapsed {time.time()-start:.0f}s', flush=True)
            dump(path, rows)
    manifest.update(status='complete', elapsed_seconds=time.time()-start)
    dump(OUT/'manifest.json', manifest)


if __name__ == '__main__':
    main()
