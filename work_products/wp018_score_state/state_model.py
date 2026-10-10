"""WP018 — a state-dependent joint scoreline distribution, as a drop-in
replacement for the independent-Poisson convolution.

WP017 showed the model's home/away outcome tilt cannot be reached by
re-weighting the training objective: given lambda_home and lambda_away the three
outcome probabilities are fully DETERMINED by convolving two independent
Poissons. WP018's diagnostic found a real mechanism that breaks that
independence — in evenly-matched fixtures the TRAILING side's scoring rate rises
~15% (1.187 -> 1.363 per 90) while the leader's barely moves.

This computes the joint distribution EXACTLY by dynamic programming over
(minute, home goals, away goals) rather than by simulation, so there is no Monte
Carlo noise to confuse with a real effect.

Two properties it must have, both asserted in tests:
  * with all state multipliers set to 1 it reproduces independent Poisson exactly;
  * the marginal expected goals are calibrated back to the input lambdas, so any
    RPS change comes from the DEPENDENCE and not from a shifted mean.
"""
import numpy as np

MAX_GOALS = 10
STEPS_PER_MIN = 16         # time discretisation. Converges first-order: outcome-level error
                           # vs exact Poisson is 9.0e-4 at 2 steps/min, 1.1e-4 at 16 — about
                           # 1% of the ~1.4pp tilt this work product is chasing.
MINUTES = 95


def state_multipliers(lead_mult=1.03, trail_mult=1.15, cap=2):
    """Per-state rate multipliers by ROLE, indexed by (own goals - opp goals).

    Defaults come from WP018's mismatch-controlled diagnostic: within evenly
    matched fixtures the trailing side's rate rises ~15% and the leader's ~3%.
    Level is 1.0 by construction. Beyond +/-cap the multiplier is held flat —
    exposure past a two-goal margin is thin and heavily selected.
    """
    m = {}
    for d in range(-cap, cap + 1):
        m[d] = 1.0 if d == 0 else (lead_mult if d > 0 else trail_mult)
    return m, cap


def joint_distribution(lam_h, lam_a, mult=None, max_goals=MAX_GOALS, calibrate_iters=3):
    """(n, max_goals+1, max_goals+1) joint scoreline probabilities.

    lam_h/lam_a: (n,) expected goals for each side over the whole match.
    `mult=None` gives plain independent Poisson (via the same DP, as a check).
    """
    lam_h = np.asarray(lam_h, float); lam_a = np.asarray(lam_a, float)
    n = len(lam_h); K = max_goals
    if mult is None:
        mults, cap = {0: 1.0}, 0
    else:
        mults, cap = mult

    gi, gj = np.meshgrid(np.arange(K + 1), np.arange(K + 1), indexing="ij")
    diff = np.clip(gi - gj, -cap, cap) if cap else np.zeros_like(gi)
    # multiplier for the HOME side at each cell (its own lead is gi-gj) and AWAY (gj-gi)
    mh = np.vectorize(lambda d: mults.get(int(d), 1.0))(diff).astype(float)
    ma = np.vectorize(lambda d: mults.get(int(-d), 1.0))(diff).astype(float)

    base_h, base_a = lam_h.copy(), lam_a.copy()
    steps = MINUTES * STEPS_PER_MIN
    for _ in range(max(calibrate_iters, 1)):
        P = np.zeros((n, K + 1, K + 1)); P[:, 0, 0] = 1.0
        # per-step rates: (n, K+1, K+1)
        rh = (base_h[:, None, None] / steps) * mh[None]
        ra = (base_a[:, None, None] / steps) * ma[None]
        ph = 1.0 - np.exp(-rh); pa = 1.0 - np.exp(-ra)
        stay = (1.0 - ph) * (1.0 - pa)
        for _s in range(steps):
            add_h = P * ph * (1.0 - pa)
            add_a = P * pa * (1.0 - ph)
            add_b = P * ph * pa                      # both in one step (tiny)
            P = P * stay
            P[:, 1:, :] += add_h[:, :-1, :]
            P[:, :, 1:] += add_a[:, :, :-1]
            P[:, 1:, 1:] += add_b[:, :-1, :-1]
            # mass that would leave the grid stays put (truncation), renormalised at the end
        P /= P.sum(axis=(1, 2), keepdims=True)
        if mult is None:
            break
        # calibrate base rates so marginal E[goals] matches the input lambdas
        eh = (P * gi[None]).sum(axis=(1, 2)); ea = (P * gj[None]).sum(axis=(1, 2))
        base_h *= np.where(eh > 1e-9, lam_h / eh, 1.0)
        base_a *= np.where(ea > 1e-9, lam_a / ea, 1.0)
    return P


def outcome_probs(P):
    """(n, 3) home/draw/away from a joint scoreline array."""
    ph = np.tril(P, -1).sum(axis=(1, 2))
    pd_ = np.trace(P, axis1=1, axis2=2)
    pa = np.triu(P, 1).sum(axis=(1, 2))
    out = np.stack([ph, pd_, pa], axis=1)
    return out / out.sum(axis=1, keepdims=True)
