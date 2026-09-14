"""
Kiểm thử các thuật toán quy hoạch động: Value Iteration, Policy Evaluation, Occupancy.
"""
import numpy as np
from src.envs.gridworld_mdp import make_choice_gridworld
from src.planning.dp import (
    value_iteration,
    policy_evaluation,
    compute_occupancy,
    expected_discounted_return,
)


def test_value_iteration_convergence():
    env = make_choice_gridworld(height=4, width=4)
    V_star, Q_star, pi_star = value_iteration(env)

    assert len(V_star) == 16
    assert Q_star.shape == (16, 4)
    assert len(pi_star) == 16

    # Trạng thái gần đích (14) phải có giá trị cao hơn trạng thái bắt đầu (0)
    near_goal_s = 14
    start_s = 0
    assert V_star[near_goal_s] > V_star[start_s]


def test_policy_evaluation_consistency():
    env = make_choice_gridworld(height=4, width=4)
    V_star, Q_star, pi_star = value_iteration(env)
    V_pi = policy_evaluation(env, pi_star)

    # Đánh giá chính sách tối ưu pi* phải cho giá trị xấp xỉ V*
    assert np.allclose(V_star, V_pi, atol=1e-5)


def test_occupancy_conservation():
    env = make_choice_gridworld(height=4, width=4)
    _, _, pi_star = value_iteration(env)
    d_s, d_sa = compute_occupancy(env, pi_star)

    # Tổng độ chiếm dụng trạng thái có chiết khấu chuẩn hóa phải bằng 1.0
    assert np.isclose(np.sum(d_s), 1.0, atol=1e-5)
    assert np.isclose(np.sum(d_sa), 1.0, atol=1e-5)
