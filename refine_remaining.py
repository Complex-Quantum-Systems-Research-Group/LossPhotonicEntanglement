"""Refine endpoint-adjacent symmetric maxima without repeating simplex runs."""
import hashlib
import json
from math import comb
from pathlib import Path
import numpy as np
from remaining_computations import OUT, ROOT, FIELDS, dump, symmetric_search, symmetric, scores
from thermometry import thermal_kernel, optimize_inputs


def main():
    changes, audits = [], []
    magnetization = np.concatenate([np.full(comb(10, k), (10-2*k)/2) for k in range(11)])
    for eta in (0., .25):
        with np.load(ROOT/'reports'/'local_thermometry'/f'N10_eta{eta:g}'/'kernels.npz') as saved:
            cache = {name: saved[name] for name in saved.files}
        for h in FIELDS:
            path = OUT/f'escape_eta{eta:g}_h{h:g}.json'
            rows = json.loads(path.read_text())
            max_change = 0.
            for row in rows:
                i = int(np.argmin(abs(cache['delays']-row['delay'])))
                k, dk = thermal_kernel(cache['energies']-h*magnetization, cache['branch_overlaps'][i], row['T'])
                fw, w = symmetric_search(k, dk)
                max_change = max(max_change, fw-row['F_symmetric'])
                row.update(F_symmetric=fw, w_star=w)
                if fw > row['F_full']:
                    row.update(F_full=fw, q_star=symmetric(w).tolist(), asymmetry=0.)
                row['cutoff_error'] = max(abs(float(scores(k, dk, row['q_star'], cutoff))-row['F_full']) for cutoff in (1e-15, 1e-11))
                row['escape'] = 1-fw/row['F_full'] if row['F_full'] > 1e-12 else None
            dump(path, rows)
            changes.append(dict(eta=eta, field=h, max_absolute_symmetric_improvement=max_change))
            # Independent existing optimizer, with a different set of seeds,
            # validates the worst detected escape at every field and eta.
            resolved = [r for r in rows if r['escape'] is not None and r['escape'] > max(1e-7, 1e-12/r['F_full'], r['cutoff_error']/r['F_full'])]
            worst = max(resolved or [r for r in rows if r['F_full'] > 1e-8], key=lambda r: r['escape'])
            i = int(np.argmin(abs(cache['delays']-worst['delay'])))
            k, dk = thermal_kernel(cache['energies']-h*magnetization, cache['branch_overlaps'][i], worst['T'])
            qbell = np.array([eta**2, .16, .16, eta**2])/(2*(eta**2+.16))
            check = optimize_inputs(k, dk, qbell, seeds=(107, 149, 211))
            discrepancy = abs(check['F_all_inputs_found']-worst['F_full'])/worst['F_full']
            assert discrepancy < max(1e-6, 1e-12/worst['F_full'], worst['cutoff_error']/worst['F_full']), (eta, h, discrepancy)
            audits.append(dict(eta=eta, field=h, T=worst['T'], delay=worst['delay'], relative_discrepancy=discrepancy))
            print(f'Refined eta={eta:g} h={h:g}; worst-point independent discrepancy={discrepancy:.3g}', flush=True)
    dump(OUT/'refinement_audit.json', dict(changes=changes, independent_optimizer=audits,
        source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('remaining_computations.py', 'refine_remaining.py', 'symmetric_qfi.py')}))
    inputs = {}
    for eta in (0., .25):
        folder = ROOT/'reports'/'local_thermometry'/f'N10_eta{eta:g}'
        for filename in ('configuration.json', 'kernels.npz'):
            path = folder/filename
            inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    dump(OUT/'input_sha256.json', inputs)


if __name__ == '__main__':
    main()
