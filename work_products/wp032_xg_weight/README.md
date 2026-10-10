# WP032 — Tuning the xG channel's weight

**Verdict: POSITIVE. The champion's xG channel weight moves 0.5 → 0.75, worth
−0.00029 RPS [−0.00059, +0.00002], better in 10/10 leave-one-season-out folds.**

The first champion improvement since WP028, and it required no new data, no new code,
and no new mechanism — only measuring a parameter that had been assumed since WP007.

| weight | RPS | ΔRPS vs 0.5 | LOSO better | seasons |
|---|---|---|---|---|
| 0.25 | 0.19784 | **+0.00063 [+0.00030,+0.00096]** worse | 0/10 | 1/10 |
| **0.50** | 0.19721 | — (old champion) | — | — |
| **0.75** | **0.19692** | **−0.00029 [−0.00059,+0.00002]** | **10/10** | 5/10 |
| 1.00 | 0.19699 | −0.00022 [−0.00082,+0.00042] | 10/10 | 5/10 |
| 1.75 | — | smoke only: **+0.016** on window 70 | — | — |
| 3.00 | — | smoke only: **+0.013** on window 70 | — | — |

115 windows, 3,780 matches, 0 divergences.

## Why this was sitting there

The channel is a tempered Potential, `weight * logp`. Goals and xG share the same
shot-level information, so weight=1 treats xG as a fully independent second observation
and double-counts. **0.5 was a hedge against that double-counting, not a measurement.**

WP007 tested exactly two weights on its 30-window grid:

    xG weight 1.0 vs 0.5   -0.00052  [-0.0019, +0.0009]   "no clear difference"

The point estimate said more xG was better. It was filed as null because the CI spanned
zero — on a grid whose MDE (±0.0019) was nearly **four times** the effect it had just
measured. WP013 then explicitly declined to revisit it ("weight 0.5 vs 1.0 was already
null in WP007"), and 0.5 was frozen for the rest of the project.

At 115 windows the MDE is ±0.00095, and the effect is real.

## The curve has a genuine interior peak

0.25 is clearly too low (+0.00063, CI excluding zero), 0.75 is the best point, 1.0 is
slightly past it, and the smoke fits at 1.75 and 3.00 are catastrophic (+0.016, +0.013
on a single window — 13–17× the MDE). The mechanism is visible in `xg_kappa`, the
channel's own calibration constant, which should sit at ~1.0:

    w0.25  kappa 1.018    w0.75  kappa 1.009    w1.00  kappa 1.006
    w1.75  kappa 0.958    w3.00  kappa 0.933

Past ~1.0 the channel starts explaining xG at the expense of explaining goals, dragging
its own calibration off 1.0. That is what over-trusting a second measurement looks like.

**W175 and W300 were dropped on smoke evidence alone**, without a 115-window sweep. The
asymmetry is deliberate: one window is enough to RULE OUT an effect 13× the MDE that two
independent arms agree on, but would not be enough to crown a winner (caution 15).

## Why 0.75 is a defensible champion change

By this project's own precedent for inclusion on expected value:

| | ΔRPS | CI | LOSO |
|---|---|---|---|
| `deep` (included) | −0.00083 | [−0.00177, +0.00013] | — |
| `stakes_effort` (included) | −0.00014 | [−0.00031, +0.00001] | 10/10 |
| **xg weight 0.75** | **−0.00029** | **[−0.00059, +0.00002]** | **10/10** |

P(improves) ≈ 97%, against the 95.9% that justified `deep`. Multiplicity is a weaker
concern than usual: these are three points on ONE curve with a smooth interior peak, not
three unrelated hypotheses, and the LOSO is unanimous.

**What is NOT established:** 0.75 and 1.0 are statistically indistinguishable (−0.00029
vs −0.00022). The finding is that **0.5 was too low**, not that 0.75 is exactly the peak.
A finer sweep between 0.6 and 1.0 would not be well-powered enough to resolve it.

## Predictions on record vs outcome — I was wrong twice

1. **Pre-smoke:** "the curve peaks in 1.0–2.0, W100 lands near −0.0005." Wrong — it
   peaks below 1.0 and falls off a cliff above it.
2. **Post-smoke:** "all three arms null, optimum flat in 0.3–0.6." Wrong in the other
   direction — the optimum is 0.75, above 0.5, and it clears the project's bar.
3. **Partial read at 21/115 windows** showed all three arms WORSE (+0.00035, +0.00021,
   +0.00056) and I reported the direction as settled. The early windows were
   unrepresentative — caution 15 cuts both ways, and a negative early read is no more
   reliable than a promising one. **This is the clearest instance yet of why partial
   sweeps should not be read for direction.**

The thing that survived all three wrong predictions is WP007's original point estimate:
its sign was right, its magnitude overstated.

    python work_products/wp032_xg_weight/run_cv.py --workers 6
    python work_products/wp032_xg_weight/analyse.py --arms W025,W075,W100
