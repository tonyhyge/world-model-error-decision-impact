# T1-A Preregistration — Decision-Signal Kill-Gate

**Protocol ID:** `T1A_DECISION_SIGNAL_RANKING`
**Frozen:** 2026-09-07, before any audit output was computed or read.
**Governs:** `research_3.md` §6. Decides whether direction **D2** (action-error
pessimism for offline MBRL) proceeds to its ~1850 GPU-hour campaign.

---

## 1. Question and decision rule

Does the induced-action-error signal

$$S_{\mathrm{cont}}(s) \;=\; \frac{\sigma_{\nabla_a Q}(s)}{\hat h(s) + \varepsilon_H}$$

rank **true induced action error** better than the value-space signal
$\sigma_Q$ (MOBILE-style Bellman inconsistency) and the state-space signal
$\sigma_{\mathrm{dyn}}$ (MOPO-style next-state dispersion)?

**If no, D2 has no method** and the campaign is cancelled before the budget is
spent. This document fixes every analysis choice in advance so that the verdict
cannot be produced by post-hoc selection.

---

## 2. Declared deviation from `research_3.md` §6.1

research_3 anchors T1-A to checkpoints of the reduced-compute MBPO wrapper
`src/baselines/adapters/vagram_mbpo.py`. Audit of the frozen cohort
`results/multienv_value_aware_20260827` (133 JSON records, read-only) shows that
wrapper does not learn a usable policy on two of three hosts:

| Host | MLE final return (10 seeds) | Note |
|---|---|---|
| Hopper-v4 | 215.6 | plateaus from ~20k steps; standing/falling policy |
| Walker2d-v4 | 187.1 | same regime |
| HalfCheetah-v4 | −111.8 | net backwards motion |

Every T1-A signal and every T1-A label is defined **through the critic and the
actor**. A degenerate critic makes the gate uninformative in both directions: a
failure would not indict the signal, and a pass would not transfer to D2.

**Deviation.** Hosts are instead model-free SAC agents trained by
`experiments/decision_signal/train_t1a_hosts.py` (150k env steps, 1 update/step,
standard SAC hyperparameters), with a fresh 7-member probabilistic ensemble
(5 elites, 20% hold-out, patience 10) fit to that agent's own replay buffer at
each checkpoint. This is also strictly closer to the D2 target setting — a critic
trained on a dataset plus an ensemble fit to the same dataset — than a horizon-1
Dyna wrapper is.

**Scope this deviation buys and does not buy.** It buys a non-degenerate decision
geometry. It does not make the result an offline-RL result: the buffer is an
online SAC replay, not a D4RL dataset. Transfer to D4RL is tested at G-D2-0, not
here.

No file under `submission/` or `results/multienv_value_aware_20260827` is written.

---

## 3. Frozen experimental design

**Cells.** 3 hosts (Hopper-v4, Walker2d-v4, HalfCheetah-v4) × 2 training seeds
(42, 43) × 2 checkpoints (50k, 150k env steps) = 12 cells. Two seeds are carried
because the manuscript's own CartPole detector required a model-seed sensitivity
analysis; a single fitted ensemble is not treated as sufficient.

**State pool.** 2,000 states per cell, collected *after* training by rolling out
the checkpoint's stochastic policy in the real environment. Each state stores
`(obs, qpos, qvel, traj_id)`; `traj_id` is the cluster unit for the bootstrap.

**Common value function.** All signals and all labels use the identical
deterministic soft value
$V(s') = \min_j Q_{\bar\psi_j}\!\big(s', \mu_\theta(s')\big)$
(no entropy term, no action sampling), so that any model-vs-environment
difference is attributable to dynamics and reward prediction only.

**Evaluation action.** $a_\pi = \mu_\theta(s)$, the actor mean (not a sample).

**Ensemble members.** Elite members only (5 of 7), matching what MOPO/MOBILE/MBPO
use at deployment.

---

## 4. Signals (model-side only; no environment access)

| # | Name | Definition |
|---|---|---|
| 1 | $\sigma_{\mathrm{dyn}}$ | $\big\|\operatorname{std}_k \hat s'_k(s,a_\pi) \oslash \sigma_{\text{replay}}\big\|_2$ |
| 2 | $\sigma_Q$ | $\operatorname{std}_k\big[\hat r_k + \gamma(1-\hat d_k)V(\hat s'_k)\big]$ |
| 3 | $\sigma_{\nabla_a Q}$ | $\sqrt{\tfrac{1}{K-1}\sum_k \|g_k-\bar g\|_2^2}$, $g_k=\nabla_a \hat Q_k|_{a_\pi}$ |
| 4 | $\hat h$ | $\big|\bar Q(a_\pi+\delta u)-2\bar Q(a_\pi)+\bar Q(a_\pi-\delta u)\big|/\delta^2$, $u=\bar g/\|\bar g\|$ |
| 5 | **$S_{\mathrm{cont}}$** | $\sigma_{\nabla_a Q}/(\hat h+\varepsilon_H)$, $\varepsilon_H=10^{-3}\cdot\operatorname{median}_{\text{pool}}\hat h$ |
| 6 | $\mathrm{Disag}_A^{\mathrm{cont}}$ | $\tfrac1K\sum_k\|a^*_k-\bar a^*\|_2$, $a^*_k=\arg\max_{a\in\mathcal A_M}\hat Q_k(s,a)$ |
| 7 | **`inv_curv`** *(control, added)* | $1/(\hat h+\varepsilon_H)$ — flatness alone |
| 8 | **`random`** *(control, added)* | $\mathcal U(0,1)$ — bootstrap calibration |

Frozen constants: $\delta = 0.05\times$ action range $= 0.1$; $\gamma=0.99$;
termination via `create_termination_fn(env_id)` applied to predicted next obs.

### Why controls 7 and 8 were added (not in research_3 §6.2)

$S_{\mathrm{cont}}$ divides by curvature, and the label $y_A$ is an arg-max
displacement, which is *also* large wherever $Q$ is flat in the action.
Both quantities therefore inflate in flat regions **for the same geometric
reason**, so a raw win of $S_{\mathrm{cont}}$ over $\sigma_Q$ on $y_A$ is
confounded. Without control 7 the audit cannot distinguish "ensemble
disagreement about the improvement direction is informative" from "avoid flat
states", and the latter is a much weaker and largely known idea. Control 8
calibrates the cluster bootstrap against a null signal.

---

## 5. Labels (environment access, offline)

Environment is restored to each pooled state with `set_state(qpos, qvel)`.
Action grid $\mathcal A_M(a_\pi)$: $M=32$ points $=\{a_\pi\}\cup\{a_\pi+\eta_j\}$,
$\eta_j\sim\mathcal N(0,0.1^2 I)$, clipped to $[-1,1]$, drawn from an RNG seeded
by `(env_id, seed, step, state_index)` and stored.

- $Q^{\mathrm{env}}(s,a) = r_{\mathrm{env}} + \gamma(1-d_{\mathrm{env}})V(s'_{\mathrm{env}})$
- $Q^{\mathrm{mod}}(s,a) = \tfrac1K\sum_k\big[\hat r_k+\gamma(1-\hat d_k)V(\hat s'_k)\big]$

| Label | Definition | Which signal *should* win it |
|---|---|---|
| $y_A$ **(primary)** | $\big\|\arg\max_{\mathcal A_M}Q^{\mathrm{mod}} - \arg\max_{\mathcal A_M}Q^{\mathrm{env}}\big\|_2$ | $S_{\mathrm{cont}}$ |
| $y_A^{\mathrm{bin}}$ | $\mathbf 1[y_A \ge q_{0.9}(y_A)]$ | $S_{\mathrm{cont}}$ |
| $y_B$ | $\big\|(\bar s'-s'_{\mathrm{env}})\oslash\sigma_{\text{replay}}\big\|_2$ at $a_\pi$ | $\sigma_{\mathrm{dyn}}$ |
| $y_C$ | $\big|Q^{\mathrm{mod}}(s,a_\pi)-Q^{\mathrm{env}}(s,a_\pi)\big|$ | $\sigma_Q$ |

---

## 6. Statistics

Spearman $\rho$ of every signal against $y_A,y_B,y_C$; AUROC and AUPRC against
$y_A^{\mathrm{bin}}$. Uncertainty by **trajectory-cluster bootstrap**, $B=2000$,
resampling whole trajectories with replacement within a cell. Paired contrasts
are recomputed inside each bootstrap replicate.

**Primary contrasts**
- $\Delta_1 = \rho(S_{\mathrm{cont}},y_A) - \rho(\sigma_Q,y_A)$
- $\Delta_2 = \rho(S_{\mathrm{cont}},y_A) - \rho(\sigma_{\mathrm{dyn}},y_A)$

**Secondary contrast (added, gates interpretation not continuation)**
- $\Delta_3 = \rho(S_{\mathrm{cont}},y_A) - \rho(\texttt{inv\_curv},y_A)$

---

## 7. Pass / fail, frozen

**PASS** iff, in $\ge 2$ of 3 hosts, and for $\ge 1$ checkpoint in each such host,
and for **both** training seeds of that host/checkpoint, the 95% cluster-bootstrap
CIs of $\Delta_1$ **and** $\Delta_2$ both lie strictly above 0.

**FAIL** otherwise. On FAIL: D2 is cancelled, the supervisor is told within the
same week, and the remaining options are D11 (physics oracle, where the sign is
actually available) and the analysis paper.

**Interpretation gate on $\Delta_3$ (does not by itself cancel D2).**
If $\Delta_3$'s CI does not lie above 0 in a majority of passing cells, then
$S_{\mathrm{cont}}$ is not distinguishable from a flatness heuristic. The D2
method is then redefined to the cheaper `inv_curv` or `Disag` variant and the
paper may not claim that ensemble disagreement in the improvement direction is
the operative ingredient. This must be reported whichever way it falls.

**Three-tier corroboration (reported, not gating).** The narrative of D2 §2.1
predicts $\sigma_{\mathrm{dyn}}$ ranks $y_B$ best, $\sigma_Q$ ranks $y_C$ best,
$S_{\mathrm{cont}}$ ranks $y_A$ best. Any inversion is reported verbatim.

---

## 8. Frozen constants

`N_states=2000`, `M=32`, grid radius `0.1`, `delta=0.1`, `gamma=0.99`,
`eps_H = 1e-3 * median(h)`, binary threshold `q_0.9`, `B=2000`,
hosts `{Hopper-v4, Walker2d-v4, HalfCheetah-v4}`, training seeds `{42,43}`,
checkpoints `{50000, 150000}`, elites only, pool RNG seed `= 10_000 + seed`.

No constant above may be changed after any audit output is read. Any change
creates a new protocol ID and is reported as such.

---

## 9. Artifacts

| Path | Content |
|---|---|
| `experiments/decision_signal/train_t1a_hosts.py` | host trainer |
| `experiments/decision_signal/run_t1a_signal_audit.py` | audit |
| `results/t1a_hosts/<host>_seed<k>/step_<n>/` | ensemble, critic, actor, replay, meta |
| `results/t1a_audit/` | per-cell signals, labels, bootstrap output |
| `docs/T1A_SIGNAL_AUDIT_REPORT.md` | post-hoc write-up |

---

## Amendment 1 — trajectory budget for the cluster bootstrap

**Filed:** 2026-09-07, after a machinery smoke test on a throwaway 3k-step
checkpoint and **before any host cell was audited or any real result read.**

**Defect.** Section 3 fixed the pool at 2,000 states but did not fix the number
of trajectories. A trained policy on these hosts runs 1,000-step episodes, so a
naive "2,000 consecutive states" pool spans 2-3 trajectories. The
trajectory-cluster bootstrap of section 6 then resamples 2-3 clusters, which
makes every confidence interval meaningless. The smoke run reported
`ntraj=2`.

**Correction.** The pool is collected by running complete episodes until **both**
`n_states = 2000` and `n_traj_min = 25` are satisfied, then uniformly subsampling
2,000 states while preserving trajectory membership. `N_TRAJ_MIN = 25` joins the
frozen constants of section 8.

**Scope.** This changes only the sampling of the pool. No signal, label, metric,
contrast or pass rule is altered. Both `n_traj_collected` and
`n_states_collected` are recorded per cell so the subsampling is auditable.
