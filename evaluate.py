"""Compare calibration methods on disjoint calibration and evaluation score files."""
import argparse
import json
from pathlib import Path

import numpy as np
from calibration import bank, metrics

CONDITIONS = ('base', 'cautious', 'null_base', 'null_cautious')
VARIANTS = ('original', 'paraphrase')
PRIMARY = ('cautious', 'prompt_mean', 'cautious_prior')


def load(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        key = r['id']
        scores = np.asarray(r['logp'])
        if scores.shape != (3,) or not np.isfinite(scores).all() or not np.isclose(np.exp(scores).sum(), 1):
            raise ValueError(f'Invalid three-class log probabilities: {key}')
        if r['condition'] in rows.setdefault(key, {}):
            raise ValueError(f'Duplicate condition: {key}')
        rows[key][r['condition']] = r
    if not rows or any(set(r) != set(CONDITIONS) for r in rows.values()):
        raise ValueError('Each input must contain all four scoring conditions.')
    for row in rows.values():
        for r in row.values():
            if any(r[k] != row['base'][k] for k in ('group', 'label', 'prompt_variant')):
                raise ValueError('Inconsistent metadata across conditions.')
    views = {v: [r for r in rows.values() if r['base']['prompt_variant'] == v] for v in VARIANTS}
    ids = lambda v: [r['base']['id'].rsplit('::', 1)[0] for r in views[v]]
    if not views['original'] or set(ids('original')) != set(ids('paraphrase')):
        raise ValueError('Both wordings must contain the same inputs.')
    other = dict(zip(ids('paraphrase'), views['paraphrase']))
    views['paraphrase'] = [other[key] for key in ids('original')]
    for a, b in zip(views['original'], views['paraphrase']):
        if any(a['base'][k] != b['base'][k] for k in ('group', 'label')):
            raise ValueError('Inconsistent metadata across wordings.')
    return views


def evaluate(cal, ev):
    groups = [r['base']['group'] for r in ev['original']]
    if set(groups) & {r['base']['group'] for r in cal['original']}:
        raise ValueError('Calibration and evaluation premise groups must be disjoint.')
    labels = [r['base']['label'] for r in ev['original']]
    if any(y not in (0, 1, 2) for y in labels):
        raise ValueError('Evaluation labels must be 0, 1 or 2.')
    y = np.array(labels)
    arr = lambda rows, condition: np.array([r[condition]['logp'] for r in rows])
    b, cb = ev['original'], cal['original']
    output = {}
    for variant in VARIANTS:
        c, cc = ev[variant], cal[variant]
        scores = bank(arr(b, 'base'), arr(c, 'cautious'), arr(b, 'null_base'),
                      arr(c, 'null_cautious'), arr(cb, 'base'), arr(cc, 'cautious'),
                      arr(cb, 'null_base'), arr(cc, 'null_cautious'))
        pred = {k: value.argmax(1) for k, value in scores.items()}
        output[variant] = {
            'metrics': {k: metrics(y, p, pred['base'], groups, 3) for k, p in pred.items()},
            'fixed_vs': {k: metrics(y, pred['fixed_differential'], p, groups, 3) for k, p in pred.items()}}
    output['joint_primary_pass'] = all(output[v]['fixed_vs'][k]['delta_ci_low'] > 0 for v in VARIANTS for k in PRIMARY)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output path.')
    result = evaluate(load(args.calibration), load(args.evaluation))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(args.output)


if __name__ == '__main__':
    main()
