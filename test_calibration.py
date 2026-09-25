"""CPU checks for the correction algebra and evaluation safeguards."""
import copy
import json
import tempfile
from pathlib import Path

import numpy as np
from calibration import bank
from evaluate import CONDITIONS, VARIANTS, evaluate, load


def main():
    b = np.log([[.7, .2, .1], [.1, .3, .6]])
    shift = np.array([.1, -.2, .3])
    out = bank(b, b + shift, b, b + shift, b, b + shift, b, b + shift)
    assert np.allclose(out['fixed_differential'], b)
    assert np.allclose(out['pcpc'], b)
    assert np.allclose(bank(b, b, b, b, b, b, b, b)['fixed_differential'], b)
    with tempfile.TemporaryDirectory() as tmp:
        views = []
        for role in ('calibration', 'evaluation'):
            rows = [dict(id=f'{role}-{i}::{v}', condition=c, group=f'{role}-{i}',
                         label=None if role == 'calibration' else (0, 2)[i],
                         prompt_variant=v, logp=b[i].tolist())
                    for i in range(2) for v in VARIANTS for c in CONDITIONS]
            path = Path(tmp) / f'{role}.jsonl'
            rows = rows[::2] + list(reversed(rows[1::2]))
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            views.append(load(path))
        result = evaluate(*views)
        assert result['original']['metrics']['fixed_differential']['accuracy'] == 100
        assert not result['joint_primary_pass']
        overlap = copy.deepcopy(views[1])
        overlap['original'][0]['base']['group'] = 'calibration-0'
        try:
            evaluate(views[0], overlap)
        except ValueError:
            pass
        else:
            raise AssertionError('Overlapping calibration groups accepted.')
        path.write_text(json.dumps(rows[0]) + '\n')
        try:
            load(path)
        except ValueError:
            pass
        else:
            raise AssertionError('Incomplete score file accepted.')
    print('Calibration algebra, score loading and evaluation checks passed.')


if __name__ == '__main__':
    main()
