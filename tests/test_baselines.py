"""
Kiểm thử các thuật toán Baseline: Q-Learning và Dyna-Q.
"""
from src.envs.gridworld_mdp import make_choice_gridworld
from src.baselines.tabular_baselines import QLearningAgent, DynaQAgent, train_agent


def test_q_learning_and_dyna_training():
    env = make_choice_gridworld(height=3, width=3)

    q_agent = QLearningAgent(num_states=env.num_states, num_actions=env.num_actions)
    returns_q = train_agent(env, q_agent, episodes=10, max_steps_per_episode=20)
    assert len(returns_q) == 10

    dyna_agent = DynaQAgent(
        num_states=env.num_states,
        num_actions=env.num_actions,
        planning_steps=3,
        noise_level=0.1,
    )
    returns_dyna = train_agent(env, dyna_agent, episodes=10, max_steps_per_episode=20)
    assert len(returns_dyna) == 10
