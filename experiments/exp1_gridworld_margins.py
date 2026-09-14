"""Exact GridWorld diagnostics; repeated perturbations are nested in seeds."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments.common import MAGNITUDES, SEEDS, write_csv
from src.envs.gridworld_mdp import make_choice_gridworld
from src.planning.dp import value_iteration
from src.corruptions.injector import inject_perturbation
from src.corruptions.matched_pairs import generate_matched_error_pair
from src.metrics.diagnostics import compute_action_margin


def run():
    env = make_choice_gridworld()
    v, q, pi = value_iteration(env)
    margin = compute_action_margin(q)
    # A fixed threshold over states, never over test outcomes.
    threshold = np.median(margin[:-1])
    random_rows, pairs = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for s in range(24):
            a = int(pi[s])
            for magnitude in MAGNITUDES:
                for rep in range(5):
                    target = rng.dirichlet(np.ones(25))
                    direction = target - env.transitions[s, a]
                    length = np.abs(direction).sum()
                    delta = direction * min(magnitude / length, 1.)
                    corrupted = inject_perturbation(env, s, a, delta)
                    rival_q = np.delete(corrupted.q_model[s], a).max()
                    gap = float(corrupted.q_model[s, a] - rival_q)
                    model_a = int(corrupted.pi_model[s])
                    random_rows.append(dict(seed=seed, state=s, rep=rep,
                        requested_l1=magnitude, actual_l1=corrupted.error.error_l1,
                        margin=margin[s], group='narrow' if margin[s] < threshold else 'wide',
                        anchored_gap=gap, q_error=float(np.max(np.abs(corrupted.q_model[s]-q[s]))),
                        decision_flip=int(a != model_a),
                        strict_flip=int(q[s,a]-q[s,model_a] > 1e-9),
                        policy_loss=corrupted.compute_policy_loss()))
            # thesis-v2 sweeps the reflection pair over the same magnitudes as
            # the random branch. The realized L1 saturates once the requested
            # mass exceeds what the successor probabilities can actually move,
            # and that saturation is itself reported rather than hidden.
            for magnitude, rep in [(m, r) for m in MAGNITUDES for r in range(5)]:
                comp, exp = generate_matched_error_pair(env, s, a, magnitude, rng)
                row = dict(seed=seed, state=s, rep=rep, margin=margin[s],
                           requested_l1=magnitude)
                for label, c in [('down',comp), ('up',exp)]:
                    ma = int(c.pi_model[s])
                    row.update({f'l1_{label}':c.error.error_l1,
                        f'projection_{label}':float(c.error.delta_p @ v),
                        f'gap_{label}':float(c.q_model[s,a]-np.delete(c.q_model[s],a).max()),
                        f'flip_{label}':int(a != ma),
                        f'global_flips_{label}':int(np.sum(c.pi_model[:-1] != pi[:-1])),
                        f'strict_flip_{label}':int(q[s,a]-q[s,ma] > 1e-9),
                        f'loss_{label}':c.compute_policy_loss()})
                pairs.append(row)
        print(f'GridWorld seed {seed} complete', flush=True)
    write_csv('exp1_gridworld_results.csv',random_rows)
    write_csv('exp1_matched_pairs.csv',pairs)


if __name__ == '__main__':
    run()
