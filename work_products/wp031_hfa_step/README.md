# WP031 — Home advantage: a crowd STEP, or a mis-specified prior?

**Verdict: BOTH ARMS NULL. The direction is closed, and with it the backlog's last
specific lead.**

| arm | RPS | ΔRPS vs champion | LOSO |
|---|---|---|---|
| `I_effort` (champion) | 0.19721 | — | — |
| `A_crowd` (crowdless indicator) | 0.19727 | +0.00006 [−0.00002,+0.00016] | better 0/10 |
| `B_hfa_wide` (HFA prior 0.03 → 0.15) | 0.19716 | −0.00005 [−0.00020,+0.00009] | better 9/10 |

115 windows, 3,780 matches, 0 divergences. Both effect sizes are ~1/20th of the
MDE (±0.00095). These are tight nulls, not under-powered ones.

## What was being tested, and why the premise work redirected it

WP015 tested time-varying HFA as a slow **random walk**: null. The specification it never
tested is a **step** with a known external breakpoint, and the COVID crowd shift looked
like a clean natural experiment (home win rate 0.477 → 0.379 → 0.442).

The premise check (`premise.py`) measured the defect in the **champion's own
out-of-sample predictions**, as a home TILT in goals/match (>0 = under-rates home teams):

    pre-covid    n=1408   +0.0518 [-0.0320,+0.1336]   home win 0.471
    CROWDLESS    n= 434   -0.2017 [-0.3789,-0.0454]   home win 0.392   <- REAL
    post-covid   n=1938   -0.0366 [-0.1277,+0.0530]   home win 0.442

**Supported:** the crowdless window is a real, large, CI-excluding-zero defect.

**Not supported:** a permanent post-COVID step down — pre and post CIs overlap heavily.

**Demolished by the season table:**

    2016 +0.1382 | 2020 -0.2757 (crowdless) | 2022 +0.1250 | 2024 -0.2251 (FULL crowds)

2024 had full crowds and was nearly as extreme as crowdless 2020. Season-level home
advantage is volatile and **not aligned with crowd status**, so COVID is not a special
regime — it is one draw from a wide distribution. That is why the per-season HFA arm was
dropped rather than run: it is WP015's rejected drift in another coat.

## Why `A_crowd` failed — the WP030 lesson again, in a new form

The coefficient worked exactly as designed: `crowd_hfa` fitted at **−0.0992** (66% of
prior SD, sign-consistent **91/115** windows), i.e. a ~9% suppression of the home rate,
matching the measured −0.20 goals/match tilt. Correctly signed, well identified, and
mirrored in `predict.py` (unlike WP030's intactness, crowd status is a scheduling fact
known at the opening price, so withholding it would have been the bug).

**And yet it made crowdless matches WORSE: +0.00021 [+0.00002,+0.00044].** The opposite
of its entire purpose.

The explanation is WP030's lesson restated: **the latent walk was already absorbing the
crowd effect, and it absorbed it better than a pooled coefficient can.** The walk adapts
immediately and per team; a single global `crowd_hfa` needs many crowdless matches before
it is identified, and for most of the 434 test matches the training set contained few or
none of them. Replacing an adaptive, local absorber with a slowly-identified global
constant is a downgrade — so we removed the thing that was working and substituted
something weaker.

This is now the second time (WP030 depletion, WP031 crowds) that explicitly modelling a
nuisance the latents were quietly handling has made the forecast worse.

## Why `B_hfa_wide` was a hollow finding

`home_mu`'s posterior sits at **0.2100** against a prior of `Normal(0.13, 0.03)` — 2.67
prior SDs out. That looked alarming and actionable: a prior fighting the data in all 115
windows. It is not. A `Normal(0.13, 0.03)` prior against ~3,000 matches is simply
overwhelmed; the posterior was already where the likelihood wanted it. Widening to 0.15
moved it only to 0.2227 and RPS did not care (−0.00005).

**The mistake to avoid repeating:** "posterior far from prior centre" does NOT imply
"prior is distorting the fit". With this much data it only means the prior was wrong and
harmless. Check whether the posterior actually MOVES when the prior is widened before
treating it as a defect.

`B_hfa_wide` is also *worse* on crowdless matches (+0.00048), because letting `home_mu`
drift up to 0.2227 is more wrong exactly where home advantage was suppressed — a small
independent confirmation that the crowdless regime is real.

## Predictions on record vs outcome

I predicted `B_hfa_wide` was the better bet and the one I would ship. Wrong: it is a
null, and the premise that motivated it (a binding prior) was a misreading. I also
predicted `A_crowd` would gain on crowdless matches and be flat elsewhere. Wrong in the
more interesting direction: it was flat elsewhere as expected, but *worse* where it was
supposed to help.

    python work_products/wp031_hfa_step/premise.py
    python work_products/wp031_hfa_step/run_cv.py --workers 6
    python work_products/wp031_hfa_step/analyse.py
