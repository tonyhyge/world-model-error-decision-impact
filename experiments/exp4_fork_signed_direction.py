"""Signed error direction under matched severity, on a family of Fork MDPs.

Experiment 1 reflects around a randomly chosen feasible successor couple in one
fixed GridWorld. This experiment fixes the couple to the extremes of V* and
sweeps a family of 25 independently designed Fork MDPs, so the sign of the
compressive-minus-expansive contrast is estimated across layouts rather than
within one. The inference unit is the MDP.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments.common import write_csv
from src.envs.fork_mdp import make_fork_mdp
from src.planning.dp import value_iteration
from src.corruptions.matched_pairs import generate_extreme_matched_pair, repair_value
from src.metrics.diagnostics import compute_action_margin

NUM_MDPS = 25
BASE_SEED = 42
# Nine levels spanning what the Fork rows can actually absorb before the
# feasibility cap binds.
PAIR_MAGNITUDES = [.02, .05, .08, .12, .16, .20, .25, .30, .35]


def design(seed):
    """Draw one Fork layout.

    The draws are written out in sequence rather than inside a dict literal:
    the layout is a function of the draw order, so reordering these lines
    silently produces a different family of MDPs.
    """
    rng = np.random.default_rng(seed)
    branch_length = int(rng.choice([2, 3]))
    p_left = float(rng.uniform(.80, .95))
    p_right = float(np.clip(p_left - rng.uniform(.02, .12), .50, .90))
    r_left = float(rng.uniform(.9, 1.1))
    r_right = float(np.clip(r_left - rng.uniform(.02, .20), .4, 1.0))
    step_cost = float(rng.uniform(.005, .02))
    gamma = float(rng.choice([.90, .95, .98]))
    return dict(seed=seed, branch_length=branch_length, p_left=p_left,
                p_right=p_right, r_left=r_left, r_right=r_right,
                step_cost=step_cost, gamma=gamma)


def run():
    designs, rows = [], []
    for mdp_id in range(NUM_MDPS):
        spec = design(BASE_SEED + mdp_id)
        env = make_fork_mdp(branch_length=spec['branch_length'], p_left=spec['p_left'],
                            p_right=spec['p_right'], r_left=spec['r_left'],
                            r_right=spec['r_right'], step_cost=spec['step_cost'],
                            gamma=spec['gamma'])
        designs.append(dict(mdp_id=mdp_id, num_states=env.num_states, **spec))
        v, q, pi = value_iteration(env)
        margin = compute_action_margin(q)
        for state in range(env.num_states - 1):
            for action in range(env.num_actions):
                for magnitude in PAIR_MAGNITUDES:
                    pair = generate_extreme_matched_pair(env, state, action,
                                                         magnitude, v, pi)
                    if pair is None:
                        continue
                    closing, opening = pair
                    # Severity is matched by construction; record both norms so
                    # the match is checked rather than asserted.
                    c_close, c_open = repair_value(closing), repair_value(opening)
                    rows.append(dict(
                        mdp_id=mdp_id, seed=spec['seed'], state=state, action=action,
                        requested_magnitude=magnitude,
                        l1_closing=closing.error.error_l1, l1_opening=opening.error.error_l1,
                        projection_closing=float(closing.error.delta_p @ v),
                        projection_opening=float(opening.error.delta_p @ v),
                        true_margin=float(margin[state]),
                        is_greedy_action=int(action == int(pi[state])),
                        flip_closing=int(closing.pi_model[state] != pi[state]),
                        flip_opening=int(opening.pi_model[state] != pi[state]),
                        repair_closing=c_close, repair_opening=c_open,
                        repair_difference=c_close - c_open))
        print(f'Fork MDP {mdp_id} complete ({env.num_states} states)', flush=True)
    write_csv('exp4_fork_designs.csv', designs)
    write_csv('exp4_fork_matched_pairs.csv', rows)


if __name__ == '__main__':
    run()
