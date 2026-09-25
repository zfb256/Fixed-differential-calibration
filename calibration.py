"""Fixed differential calibration and the evaluated score baselines."""
from collections import defaultdict
import numpy as np


def bank(b, c, b0, c0, cb, cc, cb0, cc0):
    return {'base': b, 'cautious': c, 'prompt_mean': (b+c)/2,
            'fixed_differential': c-(cc0-cb0).mean(0), 'pcpc': c-c0+b0,
            'single_local': c-c0, 'single_fixed_mean_log': c-cc0.mean(0),
            'single_fixed_log_mean_prob': c-np.log(np.exp(cc0).mean(0)),
            'base_prior': b-np.log(np.exp(cb).mean(0)),
            'cautious_prior': c-np.log(np.exp(cc).mean(0)),
            'fixed_positive': b-np.maximum(cc0-cb0, 0).mean(0)}


def metrics(y, pred, base, groups, classes):
    correct, original = pred == y, base == y
    f1, recall = [], []
    for label in range(classes):
        tp = np.sum((pred == label) & (y == label))
        denom = np.sum(pred == label)+np.sum(y == label)
        f1.append(float(2*tp/denom) if denom else 0.)
        recall.append(float(tp/np.sum(y == label)) if np.any(y == label) else None)
    # Resample premise/claim groups, preserving correlated examples within each group.
    group_counts = defaultdict(lambda: [0, 0])
    for group, delta in zip(groups, correct.astype(int)-original.astype(int)):
        group_counts[group][0] += int(delta)
        group_counts[group][1] += 1
    values = np.array(list(group_counts.values()))
    rng = np.random.default_rng(20260921)
    draws = rng.integers(len(values), size=(2000, len(values)))
    sums = values[draws].sum(axis=1)
    ci = np.percentile(sums[:, 0]/sums[:, 1]*100, [2.5, 97.5])
    return {"n": len(y), "groups": len(values), "accuracy": float(correct.mean()*100),
            "balanced_accuracy": float(np.mean([r for r in recall if r is not None])*100),
            "macro_f1": float(np.mean(f1)*100), "recall": recall,
            "fixed": int(np.sum(correct & ~original)), "harmed": int(np.sum(~correct & original)),
            "delta_pp": float((correct.mean()-original.mean())*100),
            "delta_ci_low": float(ci[0]), "delta_ci_high": float(ci[1])}
