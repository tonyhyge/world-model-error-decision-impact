"""Regenerate summary.json, TeX tables and vector figures from committed CSVs."""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import plot_results
from experiments.common import (ROOT, MAGNITUDES, SEEDS, read_csv, mean_ci, auc,
                                cluster_bootstrap_ci)
from experiments.exp4_fork_signed_direction import PAIR_MAGNITUDES
from src.envs.gridworld_mdp import make_choice_gridworld
from src.planning.dp import value_iteration
from src.metrics.diagnostics import compute_action_margin

SUMMARY={}
BASELINE_LABELS=[('Q-learning',0.,'Q-learning'),('Dyna-Q',0.,'Dyna-Q: không thêm nhiễu'),
    ('Dyna-Q',.15,'Dyna-Q: nhiễu 15%'),('Dyna-Q',.35,'Dyna-Q: nhiễu 35%')]

def values(rows,key):return np.array([float(r[key]) for r in rows])
def subset(rows,**kw):return [r for r in rows if all(str(r[k])==str(v) for k,v in kw.items())]
def seed_mean(rows,key):return [float(values(subset(rows,seed=s),key).mean()) for s in SEEDS]


class Context:
    """Everything the figure module needs, so plotting never re-reads a CSV."""
    col=staticmethod(values)
    sub=staticmethod(subset)
    baseline_labels=BASELINE_LABELS

    def __init__(self,**kw):self.__dict__.update(kw)


def analyze(report=None):
    grid=read_csv('exp1_gridworld_results.csv'); pairs=read_csv('exp1_matched_pairs.csv')
    cart=read_csv('exp2_cartpole_results.csv'); roll=read_csv('exp2_cartpole_rollouts.csv')
    base=read_csv('exp3_baseline_results.csv')
    SUMMARY['seeds']=len(SEEDS)
    SUMMARY['grid_rows']=len(grid); SUMMARY['pair_rows']=len(pairs)
    SUMMARY['cart_rows']=len(cart)
    SUMMARY['grid_flip']=mean_ci(seed_mean(grid,'decision_flip'))
    SUMMARY['grid_strict_flip']=mean_ci(seed_mean(grid,'strict_flip'))
    SUMMARY['pair_l1_min']=float(values(pairs,'l1_down').min())
    SUMMARY['pair_l1_max']=float(values(pairs,'l1_down').max())
    SUMMARY['pair_max_l1_mismatch']=float(np.max(abs(values(pairs,'l1_down')-values(pairs,'l1_up'))))
    # Realized L1 saturates against the successor probabilities; report the ceiling.
    SUMMARY['pair_l1_achieved']={str(m):float(values(subset(pairs,requested_l1=m),'l1_down').mean())
                                 for m in MAGNITUDES}
    for metric in ['loss','flip','strict_flip']:
        for side in ['down','up']:
            SUMMARY['pair_'+metric+'_'+side]=mean_ci(seed_mean(pairs,metric+'_'+side))
        SUMMARY['pair_'+metric+'_difference']=mean_ci(np.array(seed_mean(pairs,metric+'_down'))-seed_mean(pairs,metric+'_up'))

    env=make_choice_gridworld();v,q,pi=value_iteration(env);margin=compute_action_margin(q)
    near,far,auc_margin,auc_mse=[],[],[],[]
    for seed in SEEDS:
        r=subset(cart,seed=seed);m=values(r,'margin');y=values(r,'decision_flip');e=values(r,'pred_mse')
        threshold=np.quantile(m,.3)
        near.append(y[m<threshold].mean());far.append(y[m>=threshold].mean())
        auc_margin.append(auc(y,-m));auc_mse.append(auc(y,e))
    SUMMARY['cart_flips_total']=int(values(cart,'decision_flip').sum())
    SUMMARY['cart_seeds_with_auc']=int(np.isfinite(auc_margin).sum())
    for name,data in [('cart_near',near),('cart_far',far),('cart_auc_margin',auc_margin),
                      ('cart_auc_mse',auc_mse),('cart_mse',seed_mean(cart,'pred_mse'))]:
        SUMMARY[name]=mean_ci(data)
    SUMMARY['cart_near_far_difference']=mean_ci(np.array(near)-far)
    SUMMARY['cart_auc_difference']=mean_ci(np.array(auc_margin)-auc_mse)
    for policy in ['reference','learned_model']:
        SUMMARY['cart_return_'+policy]=mean_ci(seed_mean(subset(roll,policy=policy),'return_value'))
    SUMMARY['cart_return_difference']=mean_ci(np.array(seed_mean(subset(roll,policy='learned_model'),'return_value'))-seed_mean(subset(roll,policy='reference'),'return_value'))

    for name,noise,_ in BASELINE_LABELS:
        arr=np.array([values(subset(base,algorithm=name,noise_level=noise,seed=s),'greedy_return') for s in SEEDS])
        SUMMARY[f'baseline_{name}_{noise}']=mean_ci(arr[:,-1])
    qfinal=values(subset(base,algorithm='Q-learning',noise_level=0.,real_steps=5000),'greedy_return')
    for noise in [0.,.15,.35]:
        d=values(subset(base,algorithm='Dyna-Q',noise_level=noise,real_steps=5000),'greedy_return')
        SUMMARY[f'baseline_difference_{noise}']=mean_ci(d-qfinal)
    SUMMARY['grid_margin_min']=float(margin[:-1].min())
    SUMMARY['grid_margin_median']=float(np.median(margin[:-1]))
    SUMMARY['grid_tied_states']=int((margin[:-1]<1e-9).sum())

    fork=read_csv('exp4_fork_matched_pairs.csv')
    fork_ids=sorted({int(r['mdp_id']) for r in fork})
    fork_per_mdp=[float(values(subset(fork,mdp_id=i),'repair_difference').mean()) for i in fork_ids]
    SUMMARY['fork_mdps']=len(fork_ids); SUMMARY['fork_pairs']=len(fork)
    SUMMARY['fork_difference']=cluster_bootstrap_ci(fork_per_mdp)
    SUMMARY['fork_difference_t']=mean_ci(fork_per_mdp)
    SUMMARY['fork_positive_mdps']=int(sum(1 for x in fork_per_mdp if x>1e-12))
    # Severity is matched by construction; record the residual so the claim is checked.
    SUMMARY['fork_max_l1_mismatch']=float(np.max(abs(values(fork,'l1_closing')-values(fork,'l1_opening'))))
    SUMMARY['fork_changed_pairs']=int((abs(values(fork,'repair_difference'))>1e-12).sum())
    for label,flag in [('greedy',1),('rival',0)]:
        rows_=subset(fork,is_greedy_action=flag)
        per=[float(values(subset(rows_,mdp_id=i),'repair_difference').mean()) for i in fork_ids]
        SUMMARY[f'fork_difference_{label}']=cluster_bootstrap_ci(per)

    law=read_csv('exp7_first_order_law.csv')
    pred=values(law,'predicted_shift'); meas=values(law,'measured_shift')
    ratio=values(law,'fraction_of_critical'); flip=values(law,'decision_flip')
    SUMMARY['law_rows']=len(law)
    SUMMARY['law_states']=len({int(r['state']) for r in law})
    SUMMARY['law_correlation']=float(np.corrcoef(pred,meas)[0,1])
    SUMMARY['law_median_relative_residual']=float(np.median(values(law,'relative_residual')))
    SUMMARY['law_median_corrected_residual']=float(np.median(values(law,'corrected_relative_residual')))
    SUMMARY['law_corrected_correlation']=float(np.corrcoef(values(law,'corrected_shift'),meas)[0,1])
    SUMMARY['law_max_abs_residual']=float(np.max(abs(meas-pred)))
    SUMMARY['law_threshold_agreement']=float(np.mean(values(law,'predicted_flip')==flip))
    SUMMARY['law_threshold_agreement_naive']=float(np.mean(values(law,'naive_predicted_flip')==flip))
    SUMMARY['law_flip_below']=float(flip[ratio<.9].mean())
    SUMMARY['law_flip_above']=float(flip[ratio>1.1].mean())
    naive_ratio=values(law,'fraction_of_naive_critical')
    SUMMARY['law_flip_above_naive']=float(flip[naive_ratio>1.1].mean())
    # The margin, not the action value, is what decides: it needs the rival term.
    margin_meas=values(law,'measured_margin_shift')
    SUMMARY['law_margin_correlation']=float(np.corrcoef(values(law,'predicted_margin_shift'),margin_meas)[0,1])
    SUMMARY['law_margin_correlation_naive']=float(np.corrcoef(pred,margin_meas)[0,1])
    SUMMARY['law_margin_residual']=float(np.median(
        abs(margin_meas-values(law,'predicted_margin_shift'))
        /np.maximum(abs(values(law,'predicted_margin_shift')),1e-12)))
    SUMMARY['law_margin_residual_naive']=float(np.median(
        abs(margin_meas-pred)/np.maximum(abs(pred),1e-12)))

    hold=read_csv('exp6_cartpole_holdout.csv')
    SUMMARY['holdout_states']=len(hold)
    SUMMARY['holdout_flips']=int(values(hold,'decision_flip').sum())
    SUMMARY['holdout_prevalence']=mean_ci([float(values(subset(hold,seed=s),'decision_flip').mean())
                                           for s in SEEDS])
    # Privileged scores need the reference the model is meant to replace;
    # unprivileged ones use only what a deployed agent already holds.
    HOLDOUT_SCORES={'true_margin':-1,'model_margin':-1,'sigma_dyn':1,'unprivileged_score':1}
    SUMMARY['holdout_auroc']={}
    for name,sign in HOLDOUT_SCORES.items():
        per=[auc(values(subset(hold,seed=s),'decision_flip'),
                 sign*values(subset(hold,seed=s),name)) for s in SEEDS]
        SUMMARY['holdout_auroc'][name]=mean_ci(per)
    SUMMARY['holdout_auroc_gap']=mean_ci(
        [auc(values(subset(hold,seed=s),'decision_flip'),-values(subset(hold,seed=s),'true_margin'))
         - auc(values(subset(hold,seed=s),'decision_flip'),values(subset(hold,seed=s),'unprivileged_score'))
         for s in SEEDS])

    mj=read_csv('exp5_mujoco_signal_audit.csv')
    SUMMARY['mujoco_cells']=len(mj)
    SUMMARY['mujoco_hosts']=sorted({r['env_id'] for r in mj})
    SUMMARY['mujoco_delta2_negative']=int(sum(1 for r in mj if float(r['delta2_mean'])<0))
    SUMMARY['mujoco_delta2_resolved']=int(sum(1 for r in mj if float(r['delta2_high'])<0))
    SUMMARY['mujoco_delta1_negative']=int(sum(1 for r in mj if float(r['delta1_mean'])<0))
    SUMMARY['mujoco_delta2_range']=[float(min(float(r['delta2_mean']) for r in mj)),
                                    float(max(float(r['delta2_mean']) for r in mj))]
    # The three-tier claim: each error type is ranked best by its own signal.
    SUMMARY['mujoco_best_signal']={}
    for label in ['y_A','y_B','y_C']:
        wins={}
        for r in mj:
            best=max(['sigma_dyn','sigma_Q','sigma_grad_aQ','S_cont','disag_A'],
                     key=lambda s:float(r[f'rho_{s}_{label}']))
            wins[best]=wins.get(best,0)+1
        SUMMARY['mujoco_best_signal'][label]=wins
    SUMMARY['mujoco_rho_mean']={s:{l:float(np.mean([float(r[f'rho_{s}_{l}']) for r in mj]))
                                   for l in ['y_A','y_B','y_C']}
                                for s in ['sigma_dyn','sigma_Q','sigma_grad_aQ','S_cont','disag_A']}

    plot_results.make_all(Context(grid=grid,pairs=pairs,cart=cart,roll=roll,base=base,
        fork=fork,fork_ids=fork_ids,fork_per_mdp=fork_per_mdp,mj=mj,hold=hold,
        env=env,v=v,q=q,pi=pi,margin=margin,summary=SUMMARY,
        summary_series=dict(auc_margin=auc_margin,auc_mse=auc_mse),
        episodes=len(subset(roll,seed=SEEDS[0],policy='reference')),
        optimal_return=float(base[0]['optimal_return'])),report)

    (ROOT/'results'/'summary.json').write_text(json.dumps(SUMMARY,indent=2,allow_nan=False)+'\n')
    if report:
        export(report)
    print(json.dumps(SUMMARY,indent=2))


def export(report):
    """Write the LaTeX macros and the baseline table into the report tree."""
    report=Path(report);(report/'generated').mkdir(parents=True,exist_ok=True)
    names={'cart_near':'CartNear','cart_far':'CartFar','cart_auc_margin':'CartAucMargin',
        'cart_auc_mse':'CartAucMse','cart_return_reference':'CartReturnRef',
        'cart_return_learned_model':'CartReturnModel','pair_loss_down':'PairLossDown',
        'pair_loss_up':'PairLossUp','pair_flip_down':'PairFlipDown','pair_flip_up':'PairFlipUp',
        'pair_loss_difference':'PairLossDiff','cart_return_difference':'CartReturnDiff',
        'cart_auc_difference':'CartAucDiff'}
    # The report writes decimals with a comma, so the macros must too.
    def vn(x,precision):return f'{x:.{precision}f}'.replace('.',',')
    tex=[]
    for key,macro in names.items():
        d=SUMMARY[key];precision=6 if key.startswith('pair_loss') else 3
        scale=100 if key in ['cart_near','cart_far','pair_flip_down','pair_flip_up'] else 1
        tex.append('\\newcommand{\\'+macro+'}{'+vn(d['mean']*scale,precision)+'}')
        tex.append('\\newcommand{\\'+macro+'CI}{['+vn(d['low']*scale,precision)+'; '+vn(d['high']*scale,precision)+']}')
    # Counts the prose quotes directly, so the text can never drift from the run.
    # Grouped with a dot, matching the Vietnamese convention used in the report.
    def grouped(n):return f'{n:,}'.replace(',','.')
    for macro,value in [('NumSeeds',SUMMARY['seeds']),('GridRows',SUMMARY['grid_rows']),
                        ('PairRows',SUMMARY['pair_rows']),('CartRows',SUMMARY['cart_rows']),
                        ('CartFlips',SUMMARY['cart_flips_total']),
                        ('CartSeedsWithAuc',SUMMARY['cart_seeds_with_auc']),
                        ('ForkMdps',SUMMARY['fork_mdps']),('ForkPairs',SUMMARY['fork_pairs']),
                        ('ForkPositive',SUMMARY['fork_positive_mdps']),
                        ('ForkChanged',SUMMARY['fork_changed_pairs']),
                        ('MujocoCells',SUMMARY['mujoco_cells']),
                        ('MujocoNegative',SUMMARY['mujoco_delta2_negative']),
                        ('MujocoResolved',SUMMARY['mujoco_delta2_resolved']),
                        ('HoldoutStates',SUMMARY['holdout_states']),
                        ('HoldoutFlips',SUMMARY['holdout_flips']),
                        ('LawRows',SUMMARY['law_rows']),('LawStates',SUMMARY['law_states'])]:
        tex.append('\\newcommand{\\'+macro+'}{'+grouped(value)+'}')
    # Interval-valued additions from the three imported experiments.
    for macro,key,scale,precision in [
            ('ForkDiff','fork_difference',1,5),
            ('HoldoutPrev','holdout_prevalence',100,3),
            ('HoldoutPrivAuc',None,1,3),('HoldoutUnprivAuc',None,1,3),
            ('HoldoutSigmaAuc',None,1,3),('HoldoutGap','holdout_auroc_gap',1,3)]:
        source={'HoldoutPrivAuc':SUMMARY['holdout_auroc']['true_margin'],
                'HoldoutUnprivAuc':SUMMARY['holdout_auroc']['unprivileged_score'],
                'HoldoutSigmaAuc':SUMMARY['holdout_auroc']['sigma_dyn']}.get(macro) \
               or SUMMARY[key]
        tex.append('\\newcommand{\\'+macro+'}{'+vn(source['mean']*scale,precision)+'}')
        tex.append('\\newcommand{\\'+macro+'CI}{['+vn(source['low']*scale,precision)
                   +'; '+vn(source['high']*scale,precision)+']}')
    for macro,value,precision in [('LawCorr',SUMMARY['law_correlation'],5),
                                  ('LawResidual',SUMMARY['law_median_relative_residual']*100,1),
                                  ('LawResidualCorrected',SUMMARY['law_median_corrected_residual']*100,1),
                                  ('LawAgreement',SUMMARY['law_threshold_agreement']*100,1),
                                  ('LawAgreementNaive',SUMMARY['law_threshold_agreement_naive']*100,1),
                                  ('LawFlipAboveNaive',SUMMARY['law_flip_above_naive']*100,1),
                                  ('LawMarginCorr',SUMMARY['law_margin_correlation'],5),
                                  ('LawMarginCorrNaive',SUMMARY['law_margin_correlation_naive'],3),
                                  ('LawMarginResidual',SUMMARY['law_margin_residual']*100,2),
                                  ('LawMarginResidualNaive',SUMMARY['law_margin_residual_naive']*100,1),
                                  ('LawFlipBelow',SUMMARY['law_flip_below']*100,1),
                                  ('LawFlipAbove',SUMMARY['law_flip_above']*100,1)]:
        tex.append('\\newcommand{\\'+macro+'}{'+vn(value,precision)+'}')
    (report/'generated'/'numbers.tex').write_text('\n'.join(tex)+'\n')
    rows=['\\begin{tabular}{lrr}\\toprule','Phương pháp & Trung bình $J$ & CI 95\\% \\\\ \\midrule']
    for name,noise,label in BASELINE_LABELS:
        d=SUMMARY[f'baseline_{name}_{noise}'];label=label.replace('%',r'\%')
        rows.append(f'{label} & {vn(d["mean"],3)} & [{vn(d["low"],3)}; {vn(d["high"],3)}] '+r'\\')
    rows+=['\\bottomrule\\end{tabular}']
    (report/'generated'/'baseline_table.tex').write_text('\n'.join(rows)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report');args=p.parse_args();analyze(args.report)
