"""Learned CartPole dynamics, fixed-reference decisions and paired rollouts.

The reference score is heuristic, not an optimal action-value function.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import gymnasium as gym
from experiments.common import SEEDS, write_csv
from src.envs.cartpole_continuous import CartPoleDynamics, CartPoleCompetentAgent
from src.models.ensemble_dynamics import MLPEnsembleDynamics

TEST_POINTS_PER_SEED = 2000
# Rollout episodes stay at 20: the seed is the inference unit, so interval
# width is driven by between-seed spread rather than episodes within a seed.
ROLLOUT_EPISODES = 20


def model_values(ensemble, agent, s):
    preds = [ensemble.predict(s,a)[0] for a in [0,1]]
    qs = []
    for ns in preds:
        done = abs(ns[0]) > 2.4 or abs(ns[2]) > 12*np.pi/180
        qs.append(1. if done else 1. + agent.gamma*agent.evaluate_v(ns))
    return np.array(qs), preds


def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dyn = CartPoleDynamics()
    agent = CartPoleCompetentAgent(dyn)
    rows, rollout_rows, training_rows = [], [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        data=[]
        env = gym.make('CartPole-v1')
        for ep in range(20):
            s,_ = env.reset(seed=seed*1000+ep)
            for step in range(30):
                a = int(rng.integers(2)) if rng.random()<.2 else agent.select_action(s)
                ns,r,term,trunc,_ = env.step(a)
                data.append((s.copy(),a,ns.copy()))
                s=ns
                if term or trunc: break
        states=np.array([d[0] for d in data]); actions=np.array([d[1] for d in data]); ns=np.array([d[2] for d in data])
        ensemble=MLPEnsembleDynamics(num_models=5,hidden_dim=64)
        for epoch in range(120):
            order=rng.permutation(len(data)); losses=[]
            for start in range(0,len(data),32):
                ix=order[start:start+32]
                losses.append(ensemble.train_step(states[ix],actions[ix],ns[ix]))
            training_rows.append(dict(seed=seed,epoch=epoch,n_transitions=len(data),mse=float(np.mean(losses))))
        # Test states: independent generator and declared broader uniform box.
        # thesis-v2 raises this from 400 to 2000 per seed. At the observed flip
        # rate a 400-point seed could contain no flip at all, leaving AUROC
        # undefined for that seed; 2000 points make the per-seed statistic
        # computable without touching the estimand or the decision rule.
        test_rng=np.random.default_rng(seed+200000)
        for i in range(TEST_POINTS_PER_SEED):
            s=test_rng.uniform([-0.8,-0.8,-0.12,-0.8],[0.8,0.8,0.12,0.8])
            q=agent.get_action_values(s); qhat,preds=model_values(ensemble,agent,s)
            actual=[dyn.step(s,a)[0] for a in [0,1]]
            reference_a=int(q[1]>=q[0]); model_a=int(qhat[1]>=qhat[0])
            error=np.array(preds)-actual
            rows.append(dict(seed=seed,test_id=i,margin=abs(q[1]-q[0]),
                pred_mse=float(np.mean(error**2)),q_error=float(np.max(np.abs(qhat-q))),
                decision_flip=int(reference_a!=model_a),reference_action=reference_a,
                model_action=model_a,reference_score_regret=float(q[reference_a]-q[model_a])))
        for episode in range(ROLLOUT_EPISODES):
            for name in ['reference','learned_model']:
                s,_=env.reset(seed=seed*1000+500+episode)
                total=0
                for step in range(500):
                    if name=='reference': a=agent.select_action(s)
                    else:
                        qs,_=model_values(ensemble,agent,s); a=int(qs[1]>=qs[0])
                    s,r,term,trunc,_=env.step(a); total+=r
                    if term or trunc: break
                rollout_rows.append(dict(seed=seed,episode=episode,policy=name,return_value=total))
        env.close()
        print(f'CartPole seed {seed} complete ({len(data)} training transitions)',flush=True)
    write_csv('exp2_cartpole_results.csv',rows)
    write_csv('exp2_cartpole_rollouts.csv',rollout_rows)
    write_csv('exp2_training.csv',training_rows)


if __name__=='__main__': run()
