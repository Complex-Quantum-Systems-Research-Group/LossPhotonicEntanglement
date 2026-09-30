"""Summarize completed local thermometry campaigns and plot saved QFI."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    text = ['# Local thermometry results', '',
            'Information is per incident two-photon pair for x=log(T/J). '
            'These are numerical best-found input and delay optima. '
            'No separable upper bound was computed, so positive Bell gaps '
            'do not certify a strict advantage.', '']
    compact = []
    for directory in sorted(root.glob('N*_eta*')):
        config = json.loads((directory/'configuration.json').read_text())
        if config['status'] != 'complete':
            raise ValueError(f'Incomplete campaign: {directory}')
        rows = [json.loads(line) for line in (directory/'pointwise.jsonl').read_text().splitlines()]
        refined = json.loads((directory/'delay_optimized.json').read_text())
        text += [f"## N={config['N']}, eta={config['controls']['eta1']:g}", '',
                 f"Completed {len(rows)} common-delay comparisons. "
                 f"Maximum reported absolute QFI numerical sensitivity: {max(r['numerical_error'] for r in rows):.3g}.", '',
                 '| Pointwise interpretation | Count |', '| --- | ---: |']
        text += [f'| {label} | {count} |' for label, count in config['interpretation_counts'].items()]
        text += ['', '### Independently optimized delays', '',
                 '| T/J | Bell QFI | Bell delay | Product QFI found | Product delay | Bell/product |',
                 '| ---: | ---: | ---: | ---: | ---: | ---: |']
        for entry in refined:
            b = entry['bell']['best_found']
            s = entry['separable_best_found']['best_found']
            ratio = entry['independent_delay_ratio']
            ratio_text = 'suppressed' if ratio is None else f'{ratio:.6g}'
            text.append(f"| {entry['T_over_J']:.8g} | {b['F_bell']:.8g} | {b['delay']:.7g} | {s['F_separable_found']:.8g} | {s['delay']:.7g} | {ratio_text} |")
        candidates = [r for r in rows if r['interpretation'].startswith('possible_advantage')]
        if candidates:
            text += ['', '### Possible pointwise Bell advantage', '',
                     'The following points passed the larger product grid and recorded numerical checks; '
                     'they remain comparisons against the best product found.', '',
                     '| T/J | Delay | Bell QFI | Product QFI found | Gap |',
                     '| ---: | ---: | ---: | ---: | ---: |']
            text += [f"| {r['T_over_J']:.8g} | {r['delay']:g} | {r['F_bell']:.8g} | {r['F_separable_found']:.8g} | {r['absolute_gap']:.5g} |" for r in candidates]
        statuses = [s for r in rows for s in r['optimizer_status']]
        text += ['', f"Optimizer runs without a success flag: {sum(not s['success'] for s in statuses)}/{len(statuses)}. "
                 'All candidate values and statuses are retained; a success flag is not a global-optimality certificate.', '']
        temperatures = config['temperatures']
        ncols = min(4, len(temperatures))
        nrows = (len(temperatures)+ncols-1)//ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.1*nrows), squeeze=False, constrained_layout=True)
        for ax, temperature in zip(axes.flat, temperatures):
            subset = [r for r in rows if r['T_over_J'] == temperature]
            for field, label, style in [('F_bell', 'Bell', '-'), ('F_separable_found', 'Product found', '--'),
                                        ('F_all_inputs_found', 'All inputs found', ':')]:
                ax.plot([r['delay'] for r in subset], [r[field] for r in subset], style, label=label)
            ax.set(title=f'T/J = {temperature:.5g}', xlabel='Delay (tJ/hbar)', ylabel='QFI per incident pair')
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(temperatures):]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f"Local thermometry: N={config['N']}, eta={config['controls']['eta1']:g}")
        fig.savefig(directory/'qfi_vs_delay.png', dpi=180)
        plt.close(fig)
        fields = ['N', 'T_over_J', 'delay', 'theta1', 'theta2', 'eta1', 'eta2', 'F_bell',
                  'F_separable_found', 'F_all_inputs_found', 'u_opt', 'v_opt', 'absolute_gap',
                  'ratio', 'numerical_error', 'interpretation']
        compact.extend({key: r[key] for key in fields} for r in rows)
        text += [f'![QFI versus common delay]({directory.name}/qfi_vs_delay.png)', '']
    if not compact:
        raise ValueError(f'No completed campaigns under {root}')
    with (root/'pointwise_summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compact[0]))
        writer.writeheader()
        writer.writerows(compact)
    (root/'SUMMARY.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    print(root/'SUMMARY.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path('reports/local_thermometry'))
    report(parser.parse_args().root)
