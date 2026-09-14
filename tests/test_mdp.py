"""
Kiểm thử các môi trường MDP: TabularMDP, ChoiceGridWorld, CartPoleDynamics.
"""
import numpy as np
import pytest
from src.envs.tabular_mdp import TabularMDP
from src.envs.gridworld_mdp import ChoiceGridWorldMDP, make_choice_gridworld
from src.envs.cartpole_continuous import CartPoleDynamics, CartPoleCompetentAgent


def test_tabular_mdp_validation():
    # Test valid simplex transitions
    P = np.array([
        [[0.7, 0.3], [0.5, 0.5]],
        [[0.0, 1.0], [1.0, 0.0]],
    ])
    R = np.zeros((2, 2))
    mdp = TabularMDP(num_states=2, num_actions=2, transitions=P, rewards=R)
    assert mdp.validate() is True

    # Test invalid transition (sum != 1)
    P_invalid = P.copy()
    P_invalid[0, 0, 0] = 0.9  # sum = 1.2
    mdp_invalid = TabularMDP(num_states=2, num_actions=2, transitions=P_invalid, rewards=R)
    with pytest.raises(ValueError):
        mdp_invalid.validate()


def test_choice_gridworld_creation():
    env = make_choice_gridworld(height=4, width=4)
    assert env.num_states == 16
    assert env.num_actions == 4
    assert env.validate() is True

    # Kiểm tra absorbing goal tại (3, 3) -> state 15
    goal_s = 15
    for a in range(4):
        assert np.isclose(env.transitions[goal_s, a, goal_s], 1.0)


def test_cartpole_continuous_step():
    env = CartPoleDynamics()
    s0 = np.array([0.0, 0.0, 0.05, 0.0])  # nghiêng góc nhẹ
    next_s, r, done = env.step(s0, action=1)

    assert len(next_s) == 4
    assert r == 1.0
    assert done is False

    # Test competent agent action and margin
    agent = CartPoleCompetentAgent()
    a = agent.select_action(s0)
    assert a in [0, 1]
    margin = agent.compute_margin(s0)
    assert margin >= 0.0
