# WP017 — Training on the right objective: an outcome-level likelihood term

**Status: complete. Verdict: NULL at sensible weights, badly WORSE at high weight — and the motivating hypothesis is falsified.** The home/away outcome tilt this was built to fix did **not** close; it widened slightly. That's the useful finding: the tilt is not caused by an objective mismatch, which narrows what it can be.

## The hypothesis

The model maximises **goal-count** likelihood but is scored on **H/D/A outcomes**, and those objectives barely track each other (corr of goal log-lik and RPS across 30 windows: **−0.19**). Measured symptom on 1,314 matches: home wins over-predicted by **+1.4pp**, away wins under-predicted by **−1.5pp** — despite the fitted `home_mu` (0.211) matching the data's own home:away goal log-ratio (0.204) almost exactly.

The reasoning: outcome probabilities depend on the **joint** distribution, and getting both marginal goal rates right doesn't guarantee the Poisson convolution splits probability correctly across the three outcomes. So let training trade a little goal-count fit for a correct outcome split.

Implementation: `weight · Σ log P(observed outcome | λ_home, λ_away)` as a `pm.Potential`, computed from the same scoreline grid `predict.py` uses — a test asserts the two agree exactly, so training optimises precisely what RPS scores. One scalar weight, **no new latents** (avoiding what made WP016 backfire).

## Results

| Arm | Δ vs shipping config | CI (cluster) | Verdict |
|---|---|---|---|
| `J_outcome_05` (weight 0.5) | −0.00032 | [−0.00087, +0.00025] | no clear difference |
| `J_outcome_10` (weight 1.0) | +0.00001 | [−0.00109, +0.00116] | no clear difference |
| `J_outcome_30` (weight 3.0) | **+0.01249** | **[+0.00778, +0.01774]** | **much worse** |
| *[reference] the xG channel* | *−0.00258* | *[−0.00414, −0.00095]* | *better* |

Weight 0.5's point estimate is negative in all six leave-one-season-out folds, but the CI always includes zero and the effect is ~⅛ the xG channel's. Not adoptable on this evidence.

**Weight 3.0 is a five-fold degradation** — worse than the xG channel is good. Gap to Pinnacle pre-close blows out from +0.0041 to +0.0163.

### The double-counting shows up exactly where predicted

The outcome is a deterministic function of the goals, so this term re-observes the same data at coarser granularity. The fitted walk volatilities show what that does:

| Arm | `sigma_att` | `sigma_def` |
|---|---|---|
| weight 0.5 | 0.0205 | 0.0242 |
| weight 1.0 | 0.0278 | 0.0315 |
| **weight 3.0** | **0.0730** | **0.1921** |

At weight 3.0 defensive volatility inflates **~8×**. The outcome term overwhelms the goals likelihood and the latents start chasing individual match results — textbook over-fitting, and the mechanism behind the RPS collapse.

## The hypothesis is falsified: the tilt did not close

| Arm | P(home) bias | P(away) bias |
|---|---|---|
| `S3_xgchannel_only` | +0.0137 | −0.0152 |
| weight 0.5 | +0.0146 | −0.0154 |
| weight 1.0 | +0.0153 | −0.0156 |
| weight 3.0 | +0.0188 | −0.0142 |

Every arm tilts *at least as much* as the baseline. Adding an explicit outcome-level objective — with weights spanning a 6× range, including one strong enough to wreck the model — moved the tilt in the wrong direction.

**So the tilt is not a fitting artifact, and the away-favourite asymmetry barely budges** (+0.00745 → +0.00699 → +0.00659).

## What that narrows it to

If re-weighting the objective can't produce the right H/D/A split, the likeliest remaining explanation is that **no pair of independent-Poisson rates can**. The model's three outcome probabilities aren't free parameters — given `λ_home` and `λ_away` they're fully determined by the convolution. Re-weighting only slides you along that achievable surface; if the true H/D/A point lies off it, you can't reach it by trying harder.

That points at the **bivariate structure itself**, not the fitting. Dixon-Coles is the existing patch for this and it's inadequate — one global `rho` touching four low-score cells produces a near-uniform draw-ward nudge (WP007/WP014), which cannot express a home/away tilt. A genuine fix would need a different outcome model: an ordered/multinomial model on outcomes directly, or a copula with an asymmetric dependence structure — not a re-weighted Poisson.

That is a substantially narrower and more testable statement than "the tilt is unexplained", which is where WP009 and WP015 left it.

## Caveats

* **max rhat 1.070** — the highest in the project, and 0 divergences. The outcome term makes the geometry harder. Weight 3.0 in particular is straining; I'd treat that arm's magnitude as indicative, not precise (its *direction* is unambiguous).
* The term is a **weighted composite likelihood**, so the posterior is no longer that of a well-specified generative model and its credible intervals lose their usual reading. Point predictions — what RPS scores — are unaffected. `weight=0` recovers the plain model exactly.
* **Runtime 343 minutes** for 90 tasks. The 11×11 grid per match costs ~548,000 cells per gradient evaluation, at ~302,000 evaluations per window. **`outcome_max_goals` should be 7, not 10**, for any future use — that halves the grid at a cost of ~1e-5 probability mass.

## Reproducing

```bash
python work_products/wp017_outcome_likelihood/run_cv.py --workers 4   # 90 tasks, ~5.7 HOURS
python work_products/wp017_outcome_likelihood/analyse.py
```
