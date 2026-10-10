# WP036 — Match-level common shock, and what the premise checks found instead

**Verdict: DEAD ON PREMISE, three times over, in ~30 minutes and with zero CV sweeps.
No champion change. The by-product is a real but unexploitable defect and a lesson about
which statistics are worth chasing.**

## 1. The match-level common shock: killed before building

`match_effect_prior` had sat defined-but-never-called in `priors.py`, and WP017 pointed at
"the bivariate-Poisson structure" as the remaining suspect for the home/away tilt. The
natural build is a shared per-match shock `m_i` added to both sides' log rates.

A shared ADDITIVE shock on both log rates can only induce **positive** within-match goal
correlation. So the premise is a one-line question:

    corr(resid_home, resid_away) = -0.0318 [-0.0618, +0.0003]     n=3,780

Negative point estimate, CI essentially entirely non-positive. **There is no positive
correlation to capture**, so `sigma_m` would fit at ~0 (null) or, worse, impose positive
correlation where the data has slightly negative. For reference the raw goal correlation
is −0.1410 [−0.1698, −0.1107], most of which the model's lambdas already explain.

Second strike: the shock also ADDS marginal variance, and the home margin is already
UNDER-dispersed (below). WP028's NegBinomial had already measured added margin
overdispersion as worse. Double-dead.

## 2. A mis-specified check I had to correct mid-flight

My first overdispersion test used `Var(goals) / E[lambda]` and read 1.10 / 1.15, i.e.
"overdispersed". **That test is wrong**: under a perfect Poisson model with varying
lambda, the law of total variance gives

    Var(goals) = E[lambda] + Var(lambda)

so the ratio exceeds 1 even when nothing is wrong. Against the correct benchmark:

    home: Var=1.7292  E[lam]+Var(lam)=1.8936  ratio 0.9132 [0.8624,0.9648]  UNDER-dispersed
    away: Var=1.4585  E[lam]+Var(lam)=1.4713  ratio 0.9913 [0.9269,1.0589]  Poisson-consistent

The sign flipped once the benchmark was right. The home result is REAL — CI excludes 1.0.

## 3. The structural puzzle this exposed

    theta_home = intercept + attack[h] - defence[a] + hfa[h]
    theta_away = intercept + attack[a] - defence[h]
    => Var(theta_home) = Var(theta_away) + Var(hfa)   -- home is FORCED to be >= away

and the observed lambda-variance ratio 0.3230/0.2076 = 1.556 matches
(1.571/1.264)^2 = 1.555 exactly, confirming the model gives both sides identical theta
spread with the rest being `exp()` amplification. But the data implies
Var(lam_home)=0.159 < Var(lam_away)=0.195 — home LOWER in absolute terms despite the
higher mean. **The current parameterisation cannot express that at all.**

## 4. Strength-dependent home advantage: also dead on premise

The textbook mechanism that would produce it — home advantage larger for WEAKER teams,
compressing home lambdas — tested on strength quintiles:

    bucket        n     strength   tilt
    1 weakest    756     -0.710   -0.0607
    2            756     -0.119   +0.0116
    3            756     +0.218   +0.0113
    4            756     +0.549   -0.0158
    5 strongest  756     +1.148   -0.0910

    d(tilt)/d(strength) = -0.0314 [-0.1174, +0.0542]    CI spans zero

Not supported. And the shape is **non-monotone** — the model over-rates home teams at
BOTH extremes — which is the signature of too much spread, not of a strength interaction.
That is why the linear gradient test came back null.

*(My "falsification" arm there was worthless and I should not have written it:
`tilt_away = -tilt` and away strength `= -strength`, so it is algebraically the identical
regression and had to return the same number, -0.0314.)*

## 5. So: is the over-dispersion exploitable? No — tested post hoc, no sweep

Shrink the lambda spread per side, `log lam -> mean + k*(log lam - mean)`, with the mean
taken WITHIN each window so nothing peeks across the walk-forward boundary. Applied to the
3,780 existing predictions; `k=1` reproduced the champion to −0.00000, validating the rig.

     k_home  k_away      dRPS
       1.00    1.00   +0.00000   (control)
       0.95    1.00   +0.00007
       0.90    1.00   +0.00019
       0.85    1.00   +0.00035  [+0.00005,+0.00065]
       0.70    1.00   +0.00111  [+0.00049,+0.00173]   <- the value implied by the audit
       1.05    1.00   -0.00003   (slight SHARPENING, the other way)

Shrinking hurts MONOTONICALLY, and the dispersion-implied k=0.70 is the worst row.

**THE LESSON, which generalises.** `Var(goals)` vs `E[lam]+Var(lam)` is a statement about
the marginal second moment; RPS rewards **discrimination**. Shrinking the spread makes
matches look more alike and destroys the discriminative power that earns the score.
Variance-matching trades discrimination for calibration and RPS refuses that trade. The
model is "over-confident" by a variance statistic and simultaneously fine-to-under-
confident by a proper scoring rule. **A moment-matching defect is not automatically a
scoring defect** — which is also why WP009's post-hoc sharpening netted to zero.

## What this cost, and what it saved

Three hypotheses tested and closed for ~30 minutes of compute and zero 115-window sweeps.
For comparison, WP035 spent 3h45m to reach a null that a stronger premise check would have
predicted. After five straight work packages where I promised a lead and delivered a null,
the premise-first discipline is the thing that actually worked here.

    python work_products/wp036_match_shock/premise.py      # correlation + dispersion audit
    python work_products/wp036_match_shock/premise2.py     # strength-dependent HFA
    python work_products/wp036_match_shock/shrink_test.py  # post-hoc spread shrink
