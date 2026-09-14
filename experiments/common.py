"""Shared deterministic output and statistical helpers."""
import csv
from pathlib import Path
import numpy as np
from scipy.stats import t
ROOT = Path(__file__).resolve().parents[1]
# thesis-v2 widens the seed grid from five to thirty. The seed is the unit of
# inference, so this is the only lever that narrows the reported intervals.
# The first five values repeat the thesis-v1 grid, which stays a prefix of it.
SEEDS = [11 * k for k in range(1, 31)]

# Magnitudes shared by the random-perturbation and reflection-pair branches.
MAGNITUDES = [.05, .15, .35, .7]


def write_csv(name, records):
    path = ROOT / 'results' / name
    path.parent.mkdir(exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def read_csv(name):
    with (ROOT / 'results' / name).open() as f:
        return list(csv.DictReader(f))


def mean_ci(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return dict(mean=float(x.mean()) if len(x) else None, low=None, high=None, n=len(x))
    mean = float(x.mean())
    half = float(t.ppf(.975, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x)))
    return dict(mean=mean, low=mean-half, high=mean+half, n=len(x))


def cluster_bootstrap_ci(cluster_means, replicates=10000, seed=0):
    """Percentile CI from resampling whole clusters with replacement.

    Used where the inference unit is an MDP rather than a seed: rows inside one
    MDP share a layout and are not independent, so the cluster is resampled
    whole instead of the rows.
    """
    values = np.asarray(cluster_means, float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return dict(mean=float(values.mean()) if len(values) else None,
                    low=None, high=None, n=len(values))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[draws].mean(axis=1)
    return dict(mean=float(values.mean()), low=float(np.quantile(means, .025)),
                high=float(np.quantile(means, .975)), n=len(values))


def auc(y, score):
    from scipy.stats import rankdata
    y, score = np.asarray(y), np.asarray(score)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    if not n1 or not n0:
        return float('nan')
    return float((rankdata(score)[y == 1].sum() - n1*(n1+1)/2) / (n1*n0))
