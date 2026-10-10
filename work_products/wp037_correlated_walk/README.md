# WP037 — Correlated attack/defence walks

**Verdict: NULL. +0.00001 [−0.00035, +0.00037]. The last structural lead is closed.**

115 windows, 3,780 matches, 0 divergences, 58 min.

## The premise was the strongest in the project, and it held in-model

attack and defence were INDEPENDENT AR(1) walks — no shared team-quality factor, no LKJ
or cholesky anywhere in `src/`. The prior asserted independence; the data says otherwise:

    corr(attack, defence) across teams = +0.79   (p5 +0.72, p95 +0.86 over 271 rounds)

Good sides are good at both ends. And the in-model parameter agreed:

    ad_corr fitted: median +0.6807, p5 +0.6155, p95 +0.8218, sign-consistent 115/115

So the premise was right, the parameter was strongly identified, the sign was right, and
it was consistent in every single window. **RPS did not move.**

## Why — and this is the generalisable part

Both latents are estimated from THE SAME matches, so the likelihood ALREADY links them:
a win pushes attack up and defence up together. The posterior correlation was there all
along. Putting that correlation into the PRIOR double-counts information the data had
already supplied.

**The prior was wrong AND harmless.** With ~29,000 latents informed by thousands of
matches, the likelihood overwhelms prior structure. That is the same lesson as WP031's
`home_mu` sitting 2.67 prior SDs from its centre without the prior binding, and WP034's
screen finding `rho`'s `Beta(29,1)` non-binding — generalised from prior SCALES to prior
STRUCTURE.

## The third instance of one pattern

| work package | coefficient quality | ΔRPS |
|---|---|---|
| WP030 intactness | 72–104% of prior SD, sign-consistent 114/115 | **+0.00090 WORSE** |
| WP031 crowdless | 66% of prior SD, sign-consistent 91/115 | +0.00006, worse on its target subset |
| WP037 corr(att,def) | +0.68, sign-consistent **115/115** | +0.00001 |

**A well-identified, correctly-signed, season-robust coefficient is not evidence that the
forecast improves.** Three times the model clearly "wanted" the parameter and the score
did not care or got worse. This is now the most reliable negative indicator in the
project, and it should be stated in every future prediction rather than used as support.

## What the whole project now says about where wins come from

Sorting every tested change by what it touched:

**LIKELIHOOD changes — new observations about the same matches. 3 wins, 0 regressions.**
* xG measurement channel: **−0.00258**, 6/6 LOSO (WP007/WP013) — the only established win
* deep-completions channel: −0.00083, and WP033 showed removing it is worse in 0/10 folds
* xG channel WEIGHT 0.5 → 0.75: **−0.00029**, 10/10 LOSO (WP032)

**PRIOR and LATENT-STRUCTURE changes — rearranging how existing observations are used.
0 wins.**
* sigma_att/def scale, home_adv_sd, rho prior, init_scale, season_start_sigma_mult (WP034/35)
* Elo entry prior and Elo anchor (WP025 — worked, declined on external-data grounds)
* correlated attack/defence walks (this)
* time-varying HFA, HFA step/crowd regime, per-team sigma, time decay, split defence

**ADDITIVE COVARIATES on theta — 0 wins**, mostly fitting at ~2% of prior SD: congestion,
absence, lineups, keeper, intactness (the exception that fitted large, and was WORSE).

**The pattern: only adding NEW OBSERVATIONS has ever worked.** Every change that
rearranges the use of existing observations has been null. That is consistent with
WP009's conclusion that the remaining +0.00315 is missing INFORMATION, and it is now
supported by ~25 arms rather than by one analysis.

**The implication for what to try next:** not new structure. New measurement channels —
and WP014 (npxG) and WP019 (shots) exhausted the Understat feed, so that means genuinely
new data, under the constraint that it must be available at the opening price.

    python work_products/wp037_correlated_walk/premise.py    # corr + starvation
    python work_products/wp037_correlated_walk/premise2.py
    python work_products/wp037_correlated_walk/run_cv.py --workers 6
    python work_products/wp037_correlated_walk/analyse.py
