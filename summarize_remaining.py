"""Build the follow-up report from completed local computations."""
import json
from pathlib import Path
import numpy as np
from remaining_computations import OUT, ROOT, FIELDS, scores, symmetric, dump
from thermometry import thermal_kernel
from symmetric_qfi import rational_coefficients, stationary_candidates


def main():
    lines = ['# Remaining computations: local results', '',
        'All calculations ran with the project Windows `.venv` on host `mlra`.',
        'Parameters: N=10, J=Delta=1, Gaussian sigma=1, theta=0.4; eight production temperatures and 121 delays.', '',
        '## Midpoint versus Phi+', '',
        'Strict numerical comparisons use an absolute 1e-12 tolerance. Ratios exclude Phi+ QFI <= 1e-12.', '',
        '| eta | midpoint wins / all points | loses | ties within tolerance | valid ratios | median ratio | maximum ratio |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for eta in (0., .25):
        rows = json.loads((OUT/f'midpoint_eta{eta:g}.json').read_text())
        ratios = [r['ratio'] for r in rows if r['ratio'] is not None]
        wins = sum(r['difference'] > 1e-12 for r in rows)
        loses = sum(r['difference'] < -1e-12 for r in rows)
        lines.append(f'| {eta:g} | {wins}/{len(rows)} ({wins/len(rows):.3%}) | {loses} | {len(rows)-wins-loses} | {len(ratios)} | {np.median(ratios):.7g} | {max(ratios):.7g} |')
        dump(OUT/f'midpoint_failures_eta{eta:g}.json', [r for r in rows if r['difference'] < -1e-12])
    lines += ['', 'Complete failure locations are saved in `midpoint_failures_eta*.json`; pointwise files also retain ties and suppressed ratios.', '',
        '## Escape from the symmetric family', '',
        'Each point uses three full three-parameter simplex differential-evolution searches (seeds 17, 41, 73), with the continuously refined symmetric optimum retained as a candidate. These are numerical best-found optima, not certified global maxima.',
        'Escape ratios are omitted for full QFI <= 1e-12. Detection uses max(1e-12/F_full, 1e-7, cutoff_error/F_full); 1e-7 is a conservative optimizer resolution, not a rigorous bound.', '',
        '| eta | h/J | raw maximum escape | maximum above resolution | raw-max T/J | delay | winning q asymmetry | max r_K | max r_dK |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    conclusions = []
    for eta in (0., .25):
        fields = []
        for h in FIELDS:
            rows = json.loads((OUT/f'escape_eta{eta:g}_h{h:g}.json').read_text())
            fields.append(rows)
            valid = [r for r in rows if r['escape'] is not None]
            worst = max(valid, key=lambda r: r['escape'])
            resolved = [r['escape'] for r in valid if r['escape'] > max(1e-7, 1e-12/r['F_full'], r['cutoff_error']/r['F_full'])]
            lines.append(f"| {eta:g} | {h:g} | {worst['escape']:.7g} | {max(resolved, default=0):.7g} | {worst['T']:.7g} | {worst['delay']:g} | {worst['asymmetry']:.7g} | {max(r['r_K'] for r in rows):.7g} | {max(r['r_dK'] for r in rows if r['r_dK'] is not None):.7g} |")
        noise = lambda r: max(1e-7, 1e-12/r['F_full'], r['cutoff_error']/r['F_full'])
        onset = next((h for h, rs in zip(FIELDS, fields) if any(r['escape'] is not None and r['escape'] > noise(r) for r in rs)), None)
        violations, eligible = [], 0
        for series in zip(*fields):
            if all(r['escape'] is not None for r in series):
                eligible += 1
                for a, b in zip(series, series[1:]):
                    if a['escape']-b['escape'] > max(noise(a), noise(b)):
                        violations.append(dict(T=a['T'], delay=a['delay'], h_before=a['field'], h_after=b['field'], escape_before=a['escape'], escape_after=b['escape']))
        dump(OUT/f'monotonicity_failures_eta{eta:g}.json', violations)
        conclusions.append(f'eta={eta:g}: first sampled field above resolution is {onset}; {len(set((v["T"], v["delay"]) for v in violations))}/{eligible} eligible fixed-(T,delay) curves violate monotonicity ({len(violations)} adjacent-field decreases).')
        for key in ('r_K', 'r_dK'):
            slopes = []
            for series in zip(*fields[1:4]):
                if series[0]['T'] > .02 and all(r[key] is not None and r[key] > 1e-10 for r in series):
                    slopes.append(float(np.polyfit(np.log(FIELDS[1:4]), np.log([r[key] for r in series]), 1)[0]))
            conclusions.append(f'eta={eta:g}: small-field {key} log-log slope over h=0.01,0.03,0.1: median {np.median(slopes):.6g}, range [{min(slopes):.6g}, {max(slopes):.6g}] (T/J>0.02, nonnegligible norms).')
        restarts = [restart for rs in fields for row in rs for restart in row['restarts']]
        spreads = [(max(r['value'] for r in row['restarts'])-min(r['value'] for r in row['restarts']))/row['F_full'] for rs in fields for row in rs if row['restarts']]
        conclusions.append(f'eta={eta:g}: {sum(not r["success"] for r in restarts)}/{len(restarts)} DE restarts hit their iteration limit; maximum relative restart spread {max(spreads):.6g}. Individual statuses and values are retained in every pointwise row.')
    lines += [''] + conclusions + ['', 'The kernel-norm scaling is measured directly; a quadratic field premise must not be assumed.', '',
        '## Rational fit and stationary-point reduction', '',
        'Fits use 41 training points and 80 interlaced held-out points in the interior. Errors are absolute errors divided by the maximum training QFI; negligible-information points are excluded.', '',
        '| eta | maximum held-out error, degree 2/2 | degree 3/3 | exact block formula error | maximum optimum error | singular points handled spectrally |',
        '|---:|---:|---:|---:|---:|---:|']
    for eta in (0., .25):
        fits = json.loads((OUT/f'rational_fits_eta{eta:g}.json').read_text())
        folder = ROOT/'reports'/'local_thermometry'/f'N10_eta{eta:g}'
        with np.load(folder/'kernels.npz') as saved:
            cache = {name: saved[name] for name in saved.files}
        base = json.loads((OUT/f'escape_eta{eta:g}_h0.json').read_text())
        max_formula, max_opt, skipped = 0., 0., 0
        for row in base:
            idx = int(np.argmin(abs(cache['delays']-row['delay'])))
            k, dk = thermal_kernel(cache['energies'], cache['branch_overlaps'][idx], row['T'])
            if row['F_full'] <= 1e-12:
                continue
            try:
                p, q = rational_coefficients(k, dk)
                candidates = stationary_candidates(k, dk)
            except ValueError:
                skipped += 1
                continue
            w = np.linspace(.015, .985, 80)
            max_formula = max(max_formula, float(np.max(abs(p(w)/q(w)-scores(k, dk, symmetric(w))))/row['F_full']))
            value = float(np.max(scores(k, dk, symmetric(candidates))))
            max_opt = max(max_opt, abs(value-row['F_symmetric'])/row['F_full'])
        lines.append(f"| {eta:g} | {max(r['errors']['2'] for r in fits):.6g} | {max(r['errors']['3'] for r in fits):.6g} | {max_formula:.6g} | {max_opt:.6g} | {skipped} |")
    lines += ['', 'The two unnormalized parity blocks each contribute a quadratic numerator over a linear trace. Their sum is generically **cubic over quadratic**, so stationarity is a **quartic equation**, not the proposed quadratic. `symmetric_qfi.py` constructs those coefficients and obtains candidate real roots in (0,1); endpoints are evaluated using the spectral QFI. Singular blocks retain the spectral calculation. This supplies a finite polynomial-root reduction, not a universal quadratic formula for w_star.', '',
        '## Regression validation', '', 'Local command: `.\\.venv\\Scripts\\python.exe -m pytest tests -q -p no:cacheprovider`.',
        'Result: **68 passed**. Includes unequal-angle N=4 unitality, N=4 elliptical F(0), N=3 second-order matrix residual and Psi+ population, two-sided N=6 and N=10 parity at both ellipticities, 100 random-input symmetrizations for each ellipticity, exact block-formula checks, and an endpoint-adjacent optimum regression.',
        'Broad discovery from the repository root encountered Windows access-denied errors in historical report folders; explicit discovery in `tests` ran the complete source test suite.', '',
        'Cache reuse was validated against fresh N=10 channels at h=0 and 0.5 and delays 0, 1.7, 12 for both ellipticities. `manifest.json` records the local interpreter, host, optimizer settings, source hash and completion time.',
        'Boundary intervals were continuously refined after the sweep to remove false escape from narrow symmetric optima. `refinement_audit.json` records those corrections and independent existing-optimizer checks of each field\'s largest resolved escape using seeds 107, 149, 211. `input_sha256.json` fingerprints the production caches and configurations.']
    (OUT/'SUMMARY.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
