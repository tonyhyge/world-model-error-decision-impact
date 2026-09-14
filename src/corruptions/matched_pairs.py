"""Feasible reflection pairs with exactly matched realized L1 error.

Direction is defined by projection on V*, not by a guaranteed change in the
replanned margin. Experiments must measure the latter separately.
"""
from typing import Optional, Tuple
import numpy as np
from src.envs.tabular_mdp import TabularMDP
from src.planning.dp import expected_discounted_return, value_iteration
from src.corruptions.injector import CorruptedMDP, inject_perturbation


def generate_matched_error_pair(mdp: TabularMDP, state: int, action: int,
                                magnitude: float = 0.2,
                                rng: Optional[np.random.Generator] = None
                                ) -> Tuple[CorruptedMDP, CorruptedMDP]:
    if not 0 < magnitude <= 2:
        raise ValueError('L1 magnitude must be in (0, 2].')
    v, _, _ = value_iteration(mdp)
    p = mdp.transitions[state, action]
    support = np.flatnonzero(p > 1e-12)
    pairs = [(int(i), int(j)) for i in support for j in support if v[j] - v[i] > 1e-10]
    if not pairs:
        raise ValueError('No feasible nonzero value-directed reflection on this support.')
    if rng is None:
        low, high = max(pairs, key=lambda ij: v[ij[1]] - v[ij[0]])
    else:
        low, high = pairs[int(rng.integers(len(pairs)))]
    # Both p +/- delta must be feasible; never independently project a pair.
    mass = min(magnitude / 2, float(p[low]), float(p[high]))
    delta = np.zeros(mdp.num_states)
    delta[low], delta[high] = mass, -mass
    return (inject_perturbation(mdp, state, action, delta),
            inject_perturbation(mdp, state, action, -delta))


def generate_extreme_matched_pair(mdp: TabularMDP, state: int, action: int,
                                  magnitude: float,
                                  v: Optional[np.ndarray] = None,
                                  pi: Optional[np.ndarray] = None,
                                  headroom: float = 0.95
                                  ) -> Optional[Tuple[CorruptedMDP, CorruptedMDP]]:
    """Reflect between the lowest- and highest-value successors of (s, a).

    Unlike `generate_matched_error_pair`, which samples a feasible successor
    couple at random, this fixes the couple to the extremes of V* on the
    support, so the two branches realize the largest value spread the row
    admits.

    The branches are labelled by their effect on the *margin*, not on Q(s, a).
    Lowering Q closes the margin only when (s, a) is the greedy action; for a
    rival action it is raising Q that closes the margin. Labelling by value
    direction instead would mislabel every non-greedy row.

    Returns (closing, opening), or None when the row admits no non-trivial
    reflection.
    """
    if v is None or pi is None:
        v, _, pi = value_iteration(mdp)
    p = mdp.transitions[state, action]
    support = np.flatnonzero(p > 1e-6)
    if support.size < 2:
        return None
    ordered = support[np.argsort(v[support])]
    low, high = int(ordered[0]), int(ordered[-1])
    if v[high] <= v[low] + 1e-8:
        return None
    mass = min(float(magnitude), headroom * min(float(p[low]), float(p[high])))
    if mass < 1e-4:
        return None
    lowering = np.zeros(mdp.num_states)
    lowering[low], lowering[high] = mass, -mass
    closing = lowering if action == int(pi[state]) else -lowering
    return (inject_perturbation(mdp, state, action, closing),
            inject_perturbation(mdp, state, action, -closing))


def repair_value(corrupted: CorruptedMDP) -> float:
    """Return recovered in the true MDP by restoring the perturbed row.

    With a single localized error, restoring that row recovers the true model,
    so this is J(pi*) - J(pi_hat) evaluated in the true MDP. It is kept
    unclipped: the sign carries the comparison between two branches.
    """
    return float(expected_discounted_return(corrupted.true_mdp, corrupted.pi_true)
                 - expected_discounted_return(corrupted.true_mdp, corrupted.pi_model))
