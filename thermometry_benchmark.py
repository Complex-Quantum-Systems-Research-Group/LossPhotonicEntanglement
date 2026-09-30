"""Run separate ideal local-thermometry campaigns on the production grids."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.optimize import minimize, minimize_scalar

import config as cfg
from new_protocol import build_probe_weights
from thermometry import (BELL, CUTOFFS, ThermometryChannel, diagnose_input,
                        optimize_inputs, output_from_probabilities,
                        product_probabilities, quantum_fisher_information)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def vector_json(vector):
    return [[float(z.real), float(z.imag)] for z in vector]


def evaluate(channel, temperature, delay, controls, seeds, retained_product=None):
    k, dk = channel.kernel(temperature, delay)
    row = optimize_inputs(k, dk, channel.q_bell, seeds=seeds)
    if retained_product is not None:
        q = product_probabilities(*retained_product)
        value = quantum_fisher_information(*output_from_probabilities(q, k, dk))
        row['optimizer_status'].append({'method': 'retained_delay_candidate', 'success': True, 'value': value})
        if value > row['F_separable_found']:
            row.update(F_separable_found=value, u_opt=float(retained_product[0]), v_opt=float(retained_product[1]))
        if value > row['F_all_inputs_found']:
            row.update(F_all_inputs_found=value, q_all_opt=q.tolist())
    vectors = {'bell': BELL,
               'separable_best_found': channel.basis @ np.sqrt(product_probabilities(row['u_opt'], row['v_opt'])),
               'all_inputs_best_found': channel.basis @ np.sqrt(row['q_all_opt'])}
    validation = {name: diagnose_input(channel, temperature, delay, vector) for name, vector in vectors.items()}
    error = max(v['numerical_error'] for v in validation.values())
    gap = row['F_bell']-row['F_separable_found']
    negligible = max(1e-10, 10*error)
    ratio = row['F_bell']/row['F_separable_found'] if row['F_separable_found'] > negligible else None
    restarts = [r['value'] for r in row['optimizer_status'] if r['method'] == 'product_DE']
    spread = max(restarts)-min(restarts)
    stable = max(v['derivative_absolute_errors'][-1] for v in validation.values()) < 1e-8 and spread < max(1e-9, 10*error)
    if max(row['F_bell'], row['F_separable_found']) <= negligible:
        interpretation = 'negligible_information'
    elif gap < -max(1e-10, 10*error):
        interpretation = 'found_product_exceeds_bell'
    elif gap > max(1e-10, 10*error):
        interpretation = 'possible_advantage_stable' if stable else 'possible_advantage_unresolved_numerics'
    else:
        interpretation = 'no_resolved_gap'
    row.update(N=channel.n, T_over_J=float(temperature), delay=float(delay), **controls,
               absolute_gap=gap, ratio=ratio, qfi_cutoff=1e-13, numerical_error=error,
               selected_input_vectors_lab={name: vector_json(v) for name, v in vectors.items()},
               validation=validation, interpretation=interpretation,
               product_restart_spread=spread, candidate_stability_checks_passed=bool(stable))
    return row


def refine_delays(channel, temperature, delays, rows, controls, seeds):
    """Refine every sampled local maximum; retain endpoints and grid winners.

    Product delay and preparation are optimized jointly in each bracket.
    This is a best-found search, not a certified bound between grid points.
    """
    summaries = {}
    for name, field in [('bell', 'F_bell'), ('separable_best_found', 'F_separable_found')]:
        values = np.array([r[field] for r in rows])
        index = int(np.argmax(values))
        best_value, best_delay = float(values[index]), float(delays[index])
        best_uv = [rows[index]['u_opt'], rows[index]['v_opt']]
        logs = []
        peaks = [i for i in range(1, len(delays)-1)
                 if values[i] >= values[i-1] and values[i] >= values[i+1]
                 and (values[i] > values[i-1] or values[i] > values[i+1])]
        # Boundary maxima can have an unresolved interior maximum nearby.
        peaks = sorted(set(peaks + [index]))
        scale = max(best_value, 1e-12)
        def score(t, q):
            return quantum_fisher_information(*output_from_probabilities(q, *channel.kernel(temperature, t)))
        for i in peaks:
            lo, hi = float(delays[max(0, i-1)]), float(delays[min(len(delays)-1, i+1)])
            if hi <= lo:
                continue
            if name == 'bell':
                r = minimize_scalar(lambda t: -score(t, channel.q_bell)/scale,
                                    bounds=(lo, hi), method='bounded', options={'xatol': 1e-6})
                candidates = [(float(r.x), -float(r.fun)*scale, None)]
                logs.append({'success': bool(r.success), 'message': str(r.message), 'bracket': [lo, hi]})
            else:
                candidates = []
                starts = [(None, [delays[i], rows[i]['u_opt'], rows[i]['v_opt']])]
                for seed in seeds:
                    rng = np.random.default_rng(seed)
                    starts.append((seed, [rng.uniform(lo, hi), *rng.uniform(0, 1, 2)]))
                for seed, start in starts:
                    r = minimize(lambda z: -score(z[0], product_probabilities(*z[1:]))/scale,
                                 start, bounds=[(lo, hi), (0, 1), (0, 1)], method='L-BFGS-B',
                                 options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 150})
                    candidates.append((float(r.x[0]), -float(r.fun)*scale, r.x[1:].tolist()))
                    logs.append({'success': bool(r.success), 'message': str(r.message), 'seed': seed,
                                 'bracket': [lo, hi], 't_u_v': r.x.tolist(), 'value': -float(r.fun)*scale})
            for t, value, uv in candidates:
                if value > best_value:
                    best_value, best_delay = value, t
                    best_uv = uv
        refined = evaluate(channel, temperature, best_delay, controls, seeds,
                           retained_product=best_uv if name != 'bell' else None)
        # Retain the grid candidate if the final independent input search is lower.
        if refined[field] < rows[index][field]:
            refined = rows[index]
        summaries[name] = {'best_found': refined, 'refinement_status': logs,
                           'delay_window': [float(delays[0]), float(delays[-1])],
                           'grid_best': float(values[index])}
    bell = summaries['bell']['best_found']['F_bell']
    sep = summaries['separable_best_found']['best_found']['F_separable_found']
    summaries['independent_delay_absolute_gap'] = bell-sep
    error = max(summaries[name]['best_found']['numerical_error'] for name in ('bell', 'separable_best_found'))
    summaries['independent_delay_ratio'] = bell/sep if sep > max(1e-10, 10*error) else None
    return summaries


def run_campaign(args, eta, preflight):
    out = args.output / f'N{args.n_spins}_eta{eta:g}'
    out.mkdir(parents=True, exist_ok=True)
    weights = build_probe_weights(args.n_spins, cfg.probe_model, cfg.probe_sigma_sites)
    controls = dict(theta1=cfg.theta1, theta2=cfg.theta2, eta1=eta, eta2=eta)
    channel = ThermometryChannel(args.n_spins, weights, **controls, delta=cfg.delta,
                                 h_over_J=cfg.h_z/cfg.J, periodic=cfg.periodic)
    temperatures = np.asarray(args.temperatures if args.temperatures else cfg.temperature_list, float)
    delays = np.asarray(args.delays if args.delays else cfg.delta_t_list, float)
    if len(delays) == 0 or np.any(np.diff(delays) <= 0) or delays.min() < 0 or not np.isfinite(delays).all():
        raise ValueError('Delays must be finite, nonnegative and strictly increasing')
    if np.any(temperatures <= 0) or not np.isfinite(temperatures).all():
        raise ValueError('Temperatures must be finite and positive')
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    status = subprocess.run(['git', 'status', '--short'], capture_output=True, text=True).stdout
    sources = ['thermometry.py', 'thermometry_benchmark.py', 'sector_correlations.py', 'config.py',
               'new_protocol.py', 'operators.py', 'hamiltonians.py', 'tests/test_thermometry.py']
    hashes = {}
    for source in sources:
        data = Path(source).read_bytes()
        hashes[source] = hashlib.sha256(data).hexdigest()
        target = out / 'source_snapshot' / (source + '.txt')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    metadata = {'status': 'running', 'N': args.n_spins, 'controls': controls, 'weights': weights.tolist(),
                'delta': cfg.delta, 'h_over_J': cfg.h_z/cfg.J, 'periodic': cfg.periodic,
                'J_calibrated': cfg.J, 'probe_model': cfg.probe_model, 'probe_sigma_sites': cfg.probe_sigma_sites,
                'temperatures': temperatures.tolist(), 'delays': delays.tolist(), 'parameter': 'x=log(T/J)',
                'normalization': 'per incident pair; divide by two for per photon',
                'resources': {'photons': 2, 'interactions_per_photon': 1, 'bath': 'shared within trial; thermal reset between trials',
                              'intermediate_measurement': False, 'feedforward': False, 'ancillas': False,
                              'QFI_readout': 'arbitrary joint photon measurement', 'losses': False},
                'measurement': {'family': 'lab Pauli X,Y,Z tensor-product tomography', 'fractions': [1/9]*9,
                                'input_phases': 'actual Phi+; positive control amplitudes for found candidates',
                                'note': 'CFI evaluated for QFI-selected inputs; not a CFI input optimization'},
                'optimizer': {'seeds': args.seeds, 'grid_size': 21, 'advantage_grid_size': 81},
                'cutoffs': list(CUTOFFS), 'code_commit': revision, 'git_status': status, 'source_sha256': hashes,
                'validation': preflight, 'interpretation': 'Found inputs provide lower bounds. Strict Bell advantage requires a separable upper bound.',
                'numerical_error_definition': 'max absolute cutoff spread and finest fixed-input finite-difference QFI discrepancy; minimum 1e-12; excludes global search uncertainty'}
    write_json(out/'configuration.json', metadata)
    rows = []
    with (out/'pointwise.jsonl').open('w', encoding='utf-8') as handle:
        for i, delay in enumerate(delays):
            channel.overlaps(delay)
            for temperature in temperatures:
                row = evaluate(channel, float(temperature), float(delay), controls, tuple(args.seeds))
                rows.append(row)
                handle.write(json.dumps(row, allow_nan=False)+'\n')
                handle.flush()
            print(f'N={args.n_spins} eta={eta:g} delay {i+1}/{len(delays)} complete', flush=True)
    refined = []
    for temperature in temperatures:
        selected = [r for r in rows if r['T_over_J'] == float(temperature)]
        refined.append({'T_over_J': float(temperature), **refine_delays(channel, temperature, delays, selected, controls, tuple(args.seeds))})
        write_json(out/'delay_optimized.json', refined)
        print(f'N={args.n_spins} eta={eta:g} refined T/J={temperature:g}', flush=True)
    np.savez_compressed(out/'kernels.npz', energies=channel.energies, delays=delays,
                        branch_overlaps=np.array([channel.overlaps(t) for t in delays]), control_basis=channel.basis)
    metadata['status'] = 'complete'
    metadata['pointwise_rows'] = len(rows)
    metadata['interpretation_counts'] = {s: sum(r['interpretation'] == s for r in rows) for s in sorted({r['interpretation'] for r in rows})}
    write_json(out/'configuration.json', metadata)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-spins', type=int, default=10)
    parser.add_argument('--etas', type=float, nargs='+', default=[0., .25])
    parser.add_argument('--temperatures', type=float, nargs='+')
    parser.add_argument('--delays', type=float, nargs='+')
    parser.add_argument('--seeds', type=int, nargs='+', default=[17, 41, 73])
    parser.add_argument('--output', type=Path, default=Path('reports/local_thermometry'))
    args = parser.parse_args()
    if cfg.interaction_type != 'kerr' or cfg.bell_state != 'phi_plus' or cfg.J <= 0:
        parser.error('Requires Kerr, Phi+, and positive calibrated J')
    if len(set(args.seeds)) < 3:
        parser.error('At least three distinct optimizer seeds are required')
    validation = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_thermometry.py', '-q', '-p', 'no:cacheprovider'], capture_output=True, text=True)
    if validation.returncode:
        raise RuntimeError(validation.stdout+validation.stderr)
    preflight = {'passed': True, 'stdout': validation.stdout, 'stderr': validation.stderr}
    print(validation.stdout, flush=True)
    for eta in args.etas:
        print(f'Saved {run_campaign(args, eta, preflight)}', flush=True)


if __name__ == '__main__':
    main()
