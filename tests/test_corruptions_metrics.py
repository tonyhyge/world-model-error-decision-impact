"""
Kiểm thử tiêm sai số mô hình, tính toán Margin, Boundary Pressure và Decision Flips.
"""
import numpy as np
from src.envs.gridworld_mdp import make_choice_gridworld
from src.planning.dp import value_iteration
from src.corruptions.injector import inject_perturbation
from src.corruptions.matched_pairs import generate_matched_error_pair
from src.metrics.diagnostics import (
    compute_action_margin,
    compute_runner_up_action,
    compute_model_margin,
    compute_margin_deformation,
    compute_boundary_pressure,
    compute_decision_flip,
)


def test_margin_and_boundary_pressure():
    env = make_choice_gridworld(height=4, width=4)
    V_true, Q_true, pi_true = value_iteration(env)

    m_true = compute_action_margin(Q_true, pi_true)
    runner_up = compute_runner_up_action(Q_true, pi_true)

    assert np.all(m_true >= -1e-8)

    # Tiêm sai số mạnh vào state 0
    delta_p = np.zeros(env.num_states)
    delta_p[1] = -0.5
    delta_p[4] = 0.5
    corrupted = inject_perturbation(env, state=0, action=int(pi_true[0]), delta_p=delta_p)

    m_model = compute_model_margin(corrupted.q_model, pi_true, runner_up)
    delta_m = compute_margin_deformation(m_model, m_true)
    b_pressure = compute_boundary_pressure(delta_m, m_true)
    flips = compute_decision_flip(pi_true, corrupted.pi_model)

    assert len(b_pressure) == env.num_states
    assert len(flips) == env.num_states


def test_matched_pairs_error_isolation():
    env = make_choice_gridworld(height=4, width=4)
    _, _, pi_true = value_iteration(env)

    target_state = 0
    target_action = int(pi_true[target_state])

    comp, exp = generate_matched_error_pair(env, target_state, target_action, magnitude=0.4)

    # Hai cấu hình có cùng độ lớn sai số L1
    assert np.isclose(comp.error.error_l1, exp.error.error_l1, atol=1e-3)
