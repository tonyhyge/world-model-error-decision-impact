"""Validate provenance and scientific invariants of the saved run."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments.common import ROOT,SEEDS,read_csv

manifest=json.loads((ROOT/'results/manifest.json').read_text())
for category in ['source_sha256','result_sha256','frozen_evidence_sha256']:
    for filename,expected in manifest[category].items():
        assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==expected,filename
S=len(SEEDS)
counts={'exp1_gridworld_results.csv':S*24*4*5,'exp1_matched_pairs.csv':S*24*4*5,
        'exp2_cartpole_results.csv':S*2000,'exp2_cartpole_rollouts.csv':S*20*2,
        'exp2_training.csv':S*120,'exp3_baseline_results.csv':S*4*51}
# exp4 is indexed by MDP and exp5 by frozen cell, so neither is keyed on seed;
# exp6 varies in length with episode termination. They are checked separately.
for filename,n in counts.items():
    rows=read_csv(filename)
    assert len(rows)==n,(filename,len(rows))
    assert sorted({int(r['seed']) for r in rows})==SEEDS
for r in read_csv('exp1_gridworld_results.csv'):
    assert np.isclose(float(r['actual_l1']),float(r['requested_l1']),atol=1e-10)
    if float(r['margin'])>2*float(r['q_error'])+1e-9:
        assert int(r['decision_flip'])==0
for r in read_csv('exp1_matched_pairs.csv'):
    assert np.isclose(float(r['l1_down']),float(r['l1_up']),atol=1e-12)
    assert float(r['projection_down'])<0<float(r['projection_up'])
    assert np.isclose(float(r['projection_down']),-float(r['projection_up']),atol=1e-12)
for r in read_csv('exp2_cartpole_results.csv'):
    if float(r['margin'])>2*float(r['q_error'])+1e-9:
        assert int(r['decision_flip'])==0

# Fork family: severity is matched by construction and the two branches must
# project with opposite sign onto V*.
fork=read_csv('exp4_fork_matched_pairs.csv')
assert len({int(r['mdp_id']) for r in fork})==25,'expected 25 Fork MDPs'
for r in fork:
    assert np.isclose(float(r['l1_closing']),float(r['l1_opening']),atol=1e-12)
    assert float(r['projection_closing'])*float(r['projection_opening'])<0

# MuJoCo audit: the recomputation must equal the frozen GPU evidence exactly,
# otherwise the analysis has drifted away from the artifact it cites.
mj=read_csv('exp5_mujoco_signal_audit.csv')
assert len(mj)==12,(len(mj),'expected 12 preregistered cells')
for r in mj:
    frozen=json.loads((ROOT/'results/mujoco_signal_audit'/
        f"{r['env_id']}_seed{r['seed']}_step{r['step']}_stats.json").read_text())
    for signal in ['sigma_dyn','sigma_Q','S_cont','disag_A']:
        for label in ['y_A','y_B','y_C']:
            assert np.isclose(float(r[f'rho_{signal}_{label}']),
                              frozen[f'rho__{signal}'][label],atol=1e-12),(r['env_id'],signal,label)

# Held-out CartPole: the flip label must agree with the two recorded actions.
hold=read_csv('exp6_cartpole_holdout.csv')
assert sorted({int(r['seed']) for r in hold})==SEEDS
for r in hold:
    assert int(r['decision_flip'])==int(r['reference_action']!=r['model_action'])

# First-order law: the prediction must track the exact solve, and the analytic
# threshold must bracket where the greedy action actually moves.
law=read_csv('exp7_first_order_law.csv')
pred=np.array([float(r['predicted_shift']) for r in law])
meas=np.array([float(r['measured_shift']) for r in law])
ratio=np.array([float(r['fraction_of_critical']) for r in law])
flip=np.array([int(r['decision_flip']) for r in law])
assert np.corrcoef(pred,meas)[0,1]>0.99,'first-order law lost agreement'
assert flip[ratio<0.8].sum()==0,'flip recorded well below the analytic threshold'
assert flip[ratio>1.5].mean()>0.9,'no flip well above the analytic threshold'

print('PASS: source/data/frozen hashes, ten result files, matched errors,'
      ' margin invariants, Fork severity match, frozen MuJoCo agreement'
      ' and first-order law bracketing.')
