"""Do decision-oriented signals rank induced action error on MuJoCo hosts?

The expensive half of this study is not rerun here. Training the SAC hosts and
evaluating every signal on GPU produced a frozen per-state table, committed
under `results/mujoco_signal_audit/` (2000 states x 12 cells, ~1 MB). This
script recomputes every reported statistic from that table on CPU, so the
analysis is verifiable even though the rollouts are not.

Provenance for the frozen half, including GPU, library versions and source
checksums, is in `docs/mujoco_provenance/`. The decision rule was fixed in
`docs/mujoco_provenance/T1A_PREREGISTRATION.md` before any output was computed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from scipy.stats import spearmanr
from experiments.common import ROOT, auc, write_csv

EVIDENCE = ROOT / 'results' / 'mujoco_signal_audit'
HOSTS = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']
HOST_SEEDS = [42, 43]
CHECKPOINTS = [50000, 150000]
# sigma_dyn is state-space dispersion (MOPO-style), sigma_Q is value-space
# dispersion (MOBILE-style); S_cont is the decision-oriented candidate.
SIGNALS = ['sigma_dyn', 'sigma_Q', 'sigma_grad_aQ', 'S_cont', 'disag_A',
           'inv_curv', 'curv_h', 'random']
LABELS = ['y_A', 'y_B', 'y_C']
# Top decile of induced action error, matching the frozen protocol.
POSITIVE_QUANTILE = .90
BOOTSTRAP_REPLICATES = 2000


def cell_name(host, seed, step):
    return f'{host}_seed{seed}_step{step}'


def load_cell(host, seed, step):
    data = np.load(EVIDENCE / f'{cell_name(host, seed, step)}_raw.npz')
    return {k: data[k] for k in data.files}


def trajectory_bootstrap(cell, first, second, label, replicates, seed=0):
    """Resample whole trajectories, not states: states within one are dependent."""
    traj = cell['traj_id']
    ids = np.unique(traj)
    index = {t: np.flatnonzero(traj == t) for t in ids}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        picked = rng.choice(ids, size=len(ids), replace=True)
        rows = np.concatenate([index[t] for t in picked])
        y = cell[f'lab_{label}'][rows]
        if np.ptp(y) == 0:
            continue
        a = spearmanr(cell[f'sig_{first}'][rows], y).statistic
        b = spearmanr(cell[f'sig_{second}'][rows], y).statistic
        draws.append(a - b)
    draws = np.asarray(draws, float)
    return dict(mean=float(draws.mean()), low=float(np.quantile(draws, .025)),
                high=float(np.quantile(draws, .975)))


def run():
    rows = []
    for host in HOSTS:
        for seed in HOST_SEEDS:
            for step in CHECKPOINTS:
                cell = load_cell(host, seed, step)
                frozen = json.loads(
                    (EVIDENCE / f'{cell_name(host, seed, step)}_stats.json').read_text())
                label_a = cell['lab_y_A']
                threshold = float(np.quantile(label_a, POSITIVE_QUANTILE))
                binary = (label_a > threshold).astype(int)
                record = dict(env_id=host, seed=seed, step=step,
                              eval_return=frozen['eval_return'],
                              n_states=len(label_a),
                              n_trajectories=int(np.unique(cell['traj_id']).size),
                              positives=int(binary.sum()))
                for signal in SIGNALS:
                    values = cell[f'sig_{signal}']
                    for label in LABELS:
                        record[f'rho_{signal}_{label}'] = float(
                            spearmanr(values, cell[f'lab_{label}']).statistic)
                    record[f'auroc_{signal}'] = float(auc(binary, values))
                for name, rival in [('delta1', 'sigma_Q'), ('delta2', 'sigma_dyn')]:
                    boot = trajectory_bootstrap(cell, 'S_cont', rival, 'y_A',
                                                BOOTSTRAP_REPLICATES)
                    record[f'{name}_mean'] = boot['mean']
                    record[f'{name}_low'] = boot['low']
                    record[f'{name}_high'] = boot['high']
                    # The frozen run's own interval, kept alongside so the
                    # recomputation is checked rather than trusted.
                    key = ('delta1_Scont_minus_sigmaQ' if name == 'delta1'
                           else 'delta2_Scont_minus_sigmaDyn')
                    record[f'{name}_frozen_low'] = frozen['bootstrap'][key]['ci_lo']
                    record[f'{name}_frozen_high'] = frozen['bootstrap'][key]['ci_hi']
                rows.append(record)
                print(f'{cell_name(host, seed, step)} recomputed', flush=True)
    write_csv('exp5_mujoco_signal_audit.csv', rows)


if __name__ == '__main__':
    run()
