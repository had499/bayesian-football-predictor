# WP034 — Auditing the shrinkage priors

**Verdict: NULL. The regularisation scales were already right. One arm
(`home_adv_sd` 0.02 → 0.04) is directionally consistent at −0.00005 with a unanimous
10/10 LOSO, but that is 1/20th of the MDE and below this project's inclusion bar —
`HSD_XL` was run to decide whether it is a real gradient or aligned noise.**

| arm | parameter | setting | ΔRPS vs champion | LOSO better | seasons |
|---|---|---|---|---|---|
| `SIG_T` | `sigma_att/def` | 0.004 (half) | +0.00004 | 0/10 | 3/10 |
| `SIG_L` | `sigma_att/def` | 0.016 (double) | +0.00005 | 0/10 | 4/10 |
| `HSD_T` | `home_adv_sd` | 0.01 (half) | +0.00002 | 1/10 | 4/10 |
| `HSD_L` | `home_adv_sd` | 0.04 (double) | **−0.00005** [−0.00011,+0.00001] | **10/10** | 7/10 |

115 windows, 3,780 matches, 460 fits, 0 divergences, 212 min.

## The screen was the valuable part

The 15-minute screen (`screen.py`, 6 fits on one window) is the reusable contribution.
BACKLOG caution 21, learned in WP031: a posterior far from its prior centre does NOT mean
the prior binds — `home_mu` sits 2.8 prior SDs out and widening it moved RPS by −0.00005
because the likelihood simply dominated. So the only test that counts is **widen the
prior and see whether the posterior moves.**

    widened                 posterior shift                   1-window dRPS
    home_adv_sd x5          home_sd  0.0198 -> 0.0357 (+80%)      +0.00021
    stakes_effort_sd x3     stakes   -0.244 -> -0.344  (+41%)      +0.00020
    sigma_att/def x6        sigma_att 0.0204 -> 0.0242 (+19%)      +0.00099
    rho -> Beta(5,1)        NO MOVEMENT                            -0.00043
    home_sd x5 (CONTROL)    NO MOVEMENT   <- matches WP031, screen is calibrated

**It killed `rho` for 15 minutes instead of 3 hours.** I had stated one message earlier
that the AR(1) `Beta(29,1)` prior was the most promising remaining constant, reasoning
that it pins persistence near 0.97 and so controls how fast latents are forecast into the
test window. The screen showed the posterior does not move when the prior is freed to
`Beta(5,1)`: the ~29,000 latents identify `rho` themselves and the prior is irrelevant.

**The control worked**, which is why the rest of the screen can be trusted: `home_sd`
(known non-binding from WP031) came back non-binding.

## What the nulls actually tell us

`sigma_att/def` is **confirmed at its optimum**, which is a real result rather than an
absence of one: both halving and doubling it cost the same trivial amount (+0.00004,
+0.00005). A parameter that degrades symmetrically under perturbation is sitting at a
peak. The 0.008 scale was a good guess, despite the posterior landing at 0.0196 — 2.45×
the prior scale — which looked like the clearest "binding prior" in the model.

That is the lesson worth keeping: **the posterior moved (so the prior binds) and the
optimum still did not shift.** "Binds" and "mis-set" are different properties. The screen
correctly identifies the first and says nothing about the second.

## Predictions on record vs outcome — wrong in both families

I predicted: *"tighter beats looser in both families, with `HSD_T` the most likely single
winner."* Reasoning: all three binding priors got worse when loosened in the screen, so
these were shrinkage priors doing real work and nobody had tried tightening them.

* `sigma_att/def` — tighter did win, but by +0.00004 vs +0.00005. Both are null; calling
  that a win would be reading noise.
* `home_adv_sd` — **LOOSER won** (−0.00005 vs +0.00002). Backwards.
* `HSD_T`, my named favourite, was the second-worst arm.

What the screen's one-window ΔRPS column was actually worth: **nothing.** All three
values were ~1× MDE, i.e. inside noise, and the direction they implied was wrong for the
one parameter that moved. The screen is a good instrument for *which priors respond* and
a useless one for *which direction is better*. I flagged that risk before launching and
it is exactly what happened.

## What was deliberately not swept

`stakes_effort_sd` binds (+41% posterior shift) but the whole stakes term is worth only
−0.00030, so tuning its prior cannot yield much, and a 5-arm sweep would have been 4+
hours. Left unaudited on expected-value grounds, not because it is settled.

    python work_products/wp034_prior_audit/screen.py        # 15 min, 6 fits, one window
    python work_products/wp034_prior_audit/run_cv.py --workers 6
    python work_products/wp034_prior_audit/analyse.py
