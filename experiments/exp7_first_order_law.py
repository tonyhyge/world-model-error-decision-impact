"""Validate the first-order perturbation law against exactly solved MDPs.

Writing out the action value of the perturbed row,

    Qhat(s,a) - Q(s,a) = gamma * [ delta . Vhat + P(.|s,a) . (Vhat - V*) ],

the leading term is

    Qhat(s,a) - Q(s,a)  ~=  gamma * (delta . V*),

which is why the projection of the error onto V* is the quantity the rest of
the thesis uses to label direction.

Only one of the dropped terms is second order. `delta . (Vhat - V*)` is, but
`P(.|s,a) . (Vhat - V*)` is not: perturbing one row moves Vhat by O(delta), so
that term is first order and proportional to how often the perturbed row is
revisited. Writing c(s) for the discounted revisit weight of the intervened row
under the optimal policy gives the sharper statement

    Qhat(s,a) - Q(s,a)  =  gamma * (delta . V*) * (1 + gamma c(s))  +  O(delta^2),

and both predictions are recorded here so the refinement can be measured rather
than asserted. Because every quantity is exactly computable on a tabular MDP,
this is a numerical check, not a statistical estimate.

The crossing threshold needs one more step, and skipping it is a trap. The
decision at s is governed by the margin Q(s,a*) - Q(s,a2), not by Q(s,a*)
alone. Perturbing the row of a* moves Vhat, which moves Q(s,a2) as well even
though its own row is untouched. To first order,

    d[margin] = gamma (delta . V*) [ 1 + gamma c_{a*}(s) - gamma c_{a2}(s) ],

so the rival action contributes with the opposite sign through its own revisit
weight. In this layout that term dominates: gamma c_{a2} reaches 0.94 where
gamma c_{a*} is 0.15, and dropping it misstates the margin shift by a factor
of several. The threshold is therefore

    t_crit = m(s) / ( gamma * (V*(high) - V*(low)) * factor(s) ),
    factor(s) = 1 + gamma c_{a*}(s) - gamma c_{a2}(s).

Both the naive threshold (factor = 1) and the corrected one are recorded. The
naive version happens to look adequate here only because the three states
whose crossing is reachable within the feasibility cap are the three whose
factor is close to one.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments.common import write_csv
from src.envs.gridworld_mdp import make_choice_gridworld
from src.planning.dp import compute_q_from_v, to_policy_matrix, value_iteration
from src.corruptions.injector import inject_perturbation
from src.metrics.diagnostics import compute_action_margin, compute_runner_up_action

# The sweep runs on a regular grid of the feasible range rather than of the
# crossing threshold: in this layout the reflection cap is 0.0713 at every
# state while the threshold is three to six times larger at most of them, so a
# threshold-relative grid would leave most states with no feasible points.
CAP_FRACTIONS = np.linspace(.04, 1., 25)
# A reflection cannot move more than the smaller of the two successor masses.
FEASIBILITY_HEADROOM = .95


def run():
    env = make_choice_gridworld()
    v, q, pi = value_iteration(env)
    margin = compute_action_margin(q)
    runner_up = compute_runner_up_action(q)
    # Discounted state-visitation resolvent under the optimal policy; column s
    # gives how much weight flows back to the intervened row.
    policy = to_policy_matrix(pi, env.num_states, env.num_actions)
    transition = np.einsum('sa,sat->st', policy, env.transitions)
    resolvent = np.linalg.inv(np.eye(env.num_states) - env.gamma * transition)
    rows = []
    for state in range(env.num_states - 1):
        action = int(pi[state])
        p = env.transitions[state, action]
        support = np.flatnonzero(p > 1e-12)
        if support.size < 2:
            continue
        ordered = support[np.argsort(v[support])]
        low, high = int(ordered[0]), int(ordered[-1])
        span = float(v[high] - v[low])
        if span <= 1e-9:
            continue
        rival = int(runner_up[state])
        revisit = env.gamma * float(p @ resolvent[:, state])
        revisit_rival = env.gamma * float(env.transitions[state, rival]
                                          @ resolvent[:, state])
        # The rival row is never perturbed, but its action value still moves
        # with Vhat, so it enters the margin sensitivity with opposite sign.
        factor = 1. + revisit - revisit_rival
        naive_critical = float(margin[state]) / (env.gamma * span)
        critical = (naive_critical / factor if abs(factor) > 1e-9
                    else float('inf'))
        cap = FEASIBILITY_HEADROOM * min(float(p[low]), float(p[high]))
        for fraction in CAP_FRACTIONS:
            mass = float(fraction) * cap
            if mass < 1e-6:
                continue
            delta = np.zeros(env.num_states)
            delta[low], delta[high] = mass, -mass
            corrupted = inject_perturbation(env, state, action, delta)
            q_model = compute_q_from_v(corrupted.corrupted_mdp, corrupted.v_model)
            measured = float(q_model[state, action] - q[state, action])
            predicted = float(env.gamma * (corrupted.error.delta_p @ v))
            rows.append(dict(
                state=state, action=action, row=state // env.width,
                column=state % env.width,
                true_margin=float(margin[state]), value_span=span,
                rival_action=rival,
                naive_critical_mass=naive_critical, critical_mass=critical,
                margin_factor=factor,
                fraction_of_cap=float(fraction),
                fraction_of_naive_critical=mass / naive_critical,
                fraction_of_critical=mass / critical,
                mass=mass, actual_l1=corrupted.error.error_l1,
                revisit_weight=revisit, revisit_weight_rival=revisit_rival,
                predicted_shift=predicted,
                corrected_shift=predicted * (1. + revisit),
                measured_shift=measured,
                residual=measured - predicted,
                relative_residual=abs(measured - predicted) / max(abs(predicted), 1e-12),
                corrected_relative_residual=abs(measured - predicted * (1. + revisit))
                / max(abs(predicted * (1. + revisit)), 1e-12),
                naive_predicted_flip=int(mass >= naive_critical),
                predicted_flip=int(mass >= critical),
                measured_margin_shift=float(
                    (q_model[state, action] - q_model[state, rival])
                    - (q[state, action] - q[state, rival])),
                predicted_margin_shift=predicted * factor,
                decision_flip=int(corrupted.pi_model[state] != pi[state])))
        print(f'state {state} swept (margin {margin[state]:.4f})', flush=True)
    write_csv('exp7_first_order_law.csv', rows)


if __name__ == '__main__':
    run()
