import numpy as np
import pytest
from src.envs.gridworld_mdp import make_choice_gridworld
from src.planning.dp import value_iteration
from src.corruptions.matched_pairs import generate_matched_error_pair
from src.baselines.tabular_baselines import DynaQAgent
from src.envs.cartpole_continuous import CartPoleDynamics


def test_matched_pair_is_reflection_not_projection_artifact():
    env = make_choice_gridworld()
    _, _, pi = value_iteration(env)
    for s in range(24):
        comp, exp = generate_matched_error_pair(env, s, int(pi[s]), 0.35)
        assert comp.error.error_l1 > 0
        np.testing.assert_allclose(comp.error.delta_p, -exp.error.delta_p, atol=1e-12)
        assert comp.error.error_l1 == pytest.approx(exp.error.error_l1, abs=1e-12)


def test_dyna_terminal_planning_does_not_bootstrap():
    agent = DynaQAgent(2, 1, lr=1, gamma=0.9, planning_steps=1)
    agent.Q[1, 0] = 100
    agent.update(0, 0, 2, 1, True, np.random.default_rng(0))
    assert agent.Q[0, 0] == 2


def test_cartpole_matches_gymnasium_including_terminal_reward():
    import gymnasium as gym
    env = gym.make('CartPole-v1').unwrapped
    dyn = CartPoleDynamics()
    for state in [np.array([0.1, 0.2, 0.02, -0.1]), np.array([2.39, 2., 0., 0.])]:
        for a in [0, 1]:
            env.reset(seed=0)
            env.state = state.copy()
            ns, r, term, _, _ = env.step(a)
            actual, reward, done = dyn.step(state, a)
            np.testing.assert_allclose(actual, ns, atol=1e-7)
            assert reward == r
            assert done == term
    env.close()
