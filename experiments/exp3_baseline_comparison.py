"""Fixed real-interaction budget; exact greedy-policy evaluation in the true MDP."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments.common import SEEDS,write_csv
from src.envs.gridworld_mdp import make_choice_gridworld
from src.planning.dp import expected_discounted_return, value_iteration
from src.baselines.tabular_baselines import QLearningAgent,DynaQAgent


def run():
    env=make_choice_gridworld(height=4,width=4)
    _,_,optimal=value_iteration(env)
    optimal_return=expected_discounted_return(env,optimal)
    records=[]
    for seed in SEEDS:
        for name,noise in [('Q-learning',0.),('Dyna-Q',0.),('Dyna-Q',.15),('Dyna-Q',.35)]:
            rng=np.random.default_rng(seed)
            planning_rng=np.random.default_rng(seed+100000)
            agent=(QLearningAgent(16,4,lr=.15,gamma=env.gamma,epsilon=.15) if name=='Q-learning'
                   else DynaQAgent(16,4,lr=.15,gamma=env.gamma,epsilon=.15,planning_steps=5,noise_level=noise))
            state=0; episode_steps=0
            for step in range(5001):
                if step%100==0:
                    pi=agent.Q.argmax(axis=1)
                    j=expected_discounted_return(env,pi)
                    records.append(dict(seed=seed,algorithm=name,noise_level=noise,
                        real_steps=step,planning_updates=step*(5 if name=='Dyna-Q' else 0),
                        greedy_return=j,optimal_return=optimal_return,policy_loss=optimal_return-j))
                if step==5000: break
                a=agent.select_action(state,rng)
                ns=int(rng.choice(16,p=env.transitions[state,a]))
                reward=float(env.rewards[state,a])
                done=ns==env.goal_state
                if name=='Dyna-Q': agent.update(state,a,reward,ns,done,planning_rng)
                else: agent.update(state,a,reward,ns,done)
                episode_steps+=1
                state=0 if done or episode_steps>=100 else ns
                if state==0: episode_steps=0
            print(f'Baseline seed {seed} {name} noise={noise} complete',flush=True)
    write_csv('exp3_baseline_results.csv',records)


if __name__=='__main__': run()
