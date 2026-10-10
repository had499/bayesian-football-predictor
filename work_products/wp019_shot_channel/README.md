# WP019 — Shot count as a third measurement channel

**Status: complete. Verdict: NULL, with a mechanism that explains the size and closes the direction.** Adding a shot-count channel on top of the xG channel is worth **+0.00016 RPS [−0.00156, +0.00192]**. Shot count *instead of* xG is worth +0.00167 [−0.00040, +0.00377] — i.e. it is a **weaker** channel than xG. Increasing its weight monotonically makes things worse.

**The shipping config is unchanged: `L0_baseline` + `use_xg_channel=True`** (WP013).

## The premise, and why it was the wrong premise

Regressing a team-season's second-half goals/match on its first-half log shot **volume** and log shot **quality** separately gives coefficients **+1.028** and **+0.554** — difference +0.474, team-clustered CI [+0.132, +0.771]. Total xG is volume × quality, so a channel observing xG alone constrains only their product and **forces those two coefficients to be equal**. Volume is also the more repeatable skill (first→second-half r = +0.770 vs +0.567 for quality).

That coefficient claim is correct and survives scrutiny. The inference from it did not, for two reasons:

1. **I justified it with an in-sample R² gain**, quoted as "+0.0192, CI [+0.0017, +0.0503], excludes zero". Out-of-sample (leave-one-season-out) the same comparison is **+0.0244 [−0.0056, +0.0565] — null**. See the caution below; for *nested* models this kind of CI cannot exclude zero even in principle.
2. **It measured the wrong level of aggregation.** A channel's job is to infer a team's rate from **one match**. The premise check measured half-season aggregates, where 19 matches of shot counts average out into a stable quantity. Per match, a raw count discards the quality of every shot, and that is exactly the information xG supplies.

## The properly powered version of the premise check

Predicting a team's future goal rate from a **single match's** observation, n=7,680, leave-one-season-out, team-clustered CI:

| | OOS R² | incremental |
|---|---|---|
| goals alone | 0.0488 | |
| shot count alone | 0.0671 | |
| match xG alone | **0.0882** | |
| xG added to goals | | **+0.04397 [+0.01016, +0.08946] REAL** |
| shot count added to xG | | +0.00658 [−0.00193, +0.01680] null |
| shot count added to goals+xG | | +0.00845 [−0.00156, +0.02030] null |

**xG is the better per-match measurement, and it is better at the aggregate level too** (r = +0.7258 vs +0.7061). Shot count carries ~15% of the incremental information xG did. Since the xG channel is worth −0.00258 RPS, a channel 15% as informative predicts about **−0.0004** — inside this sweep's MDE of ±0.00174. **The CV null is the size the mechanism predicts**, not merely an absence of power.

## Design

One observation channel, two scalars, **zero new latents** — deliberately the xG-channel shape and not WP016's, which split a latent (~14,600 extra parameters) and made the model measurably worse.

`shots ~ NegBinomial(mu = shot_kappa · lambda, alpha = shot_alpha)`, as a tempered `pm.Potential`. NegBinomial not Poisson: shots/match are overdispersed, var/mean **2.35**. Reads `lambda_home/away`, not the suppression-only rate the xG channel uses under `split_defence` — a shot count has no shot-stopping stage in it.

| Arm | Channels | Isolates |
|---|---|---|
| `S3_xgchannel_only` (WP013) | goals + xG | the shipping baseline |
| **`E_shots_added`** | goals + xG + shots (w=0.5) | the hypothesis |
| **`F_shots_only`** | goals + shots (w=0.5) | which channel carries it (the control) |
| **`G_shots_heavy`** | goals + xG + shots (w=1.0) | is 0.5 undertrusting it? |

`F` is what makes `E` interpretable (BACKLOG caution 2).

## Results

30 windows, 1,314 paired matches. 0 divergences, max rhat 1.046.

| Arm | Pooled RPS | Gap vs Pinnacle pre-close |
|---|---|---|
| `L0_baseline` | 0.20739 | +0.00650 |
| `S3_xgchannel_only` | **0.20481** | **+0.00409** |
| `E_shots_added` | 0.20497 | +0.00458 |
| `F_shots_only` | 0.20648 | +0.00612 |
| `G_shots_heavy` | 0.20577 | +0.00549 |

| Comparison | ΔRPS | CI (cluster) | MDE as % of the 0.0039 gap |
|---|---|---|---|
| shots ADDED to xG (E vs A) | +0.00016 | [−0.00156, +0.00192] | 45% |
| shots INSTEAD of xG (F vs A) | +0.00167 | [−0.00040, +0.00377] | 53% |
| is shot count worth anything (F vs L0) | −0.00091 | [−0.00355, +0.00177] | 68% |
| heavier shot weight (G vs E) | +0.00081 | [−0.00016, +0.00181] | 25% |
| *[reference] the xG channel itself (A vs L0)* | *−0.00258* | *[−0.00414, −0.00095]* | *better* |

`E_shots_added` is better in only **3/6 seasons**; WP013's bar for the xG channel was 6/6.

**The weight ladder is the most informative part.** RPS degrades monotonically as the channel is trusted more: A (w=0) 0.20481 → E (w=0.5) 0.20497 → G (w=1.0) 0.20577. A channel carrying real information should improve as tempering is relaxed, at least initially. This one doesn't.

## The channel was not broken — it calibrated itself correctly

| | learned | truth in the data |
|---|---|---|
| `shot_kappa` (shots per unit scoring rate) | **9.33** | **9.39** |
| `shot_alpha` → implied conditional var/mean | 1.50 | 2.35 marginal (conditional is properly lower) |

So this is a null from a correctly-specified channel, not a bug.

## A mechanism I nearly asserted and the data refuted

I first read the inflating walk volatility as the cause — the extra density making the latents chase shot noise. The full ladder kills that:

| Arm | channels | `sigma_att` | vs baseline | `rho_att` |
|---|---|---|---|---|
| `L0_baseline` | goals | 0.0133 | 1.00x | 0.9970 |
| `S3_xgchannel_only` | goals+xG | 0.0170 | **1.28x** | 0.9915 |
| `F_shots_only` | goals+shots | 0.0099 | **0.74x** | 0.9922 |
| `E_shots_added` | goals+xG+shots(0.5) | 0.0141 | 1.06x | 0.9861 |
| `G_shots_heavy` | goals+xG+shots(1.0) | 0.0234 | 1.76x | 0.9618 |

The xG channel *also* raises `sigma_att` (1.28x) and it is the one thing that works; `F` *lowers* it and is null. So volatility inflation does not explain the failure. It does look like genuine over-flexing **at weight 1.0** specifically — 1.76x volatility with `rho` collapsing 0.997 → 0.962, i.e. much shorter latent memory — which is consistent with `G` being the worst arm.

## What else was ruled out along the way

Premise checks (out-of-sample, on top of shots + total xG) killed every other decomposition of the shot feed:

| Candidate | OOS R² gain | Why it fails |
|---|---|---|
| Big- vs small-chance xG | −0.0022 [−0.0098, +0.0051] | **log small-chance xG correlates 0.932 with log shots** — it *is* shot volume. In-sample it looked like the best idea in the project (+0.0249). |
| Open-play vs dead-ball xG | −0.0051 [−0.0181, +0.0083] | Coefficients genuinely differ (+0.692 [+0.451, +0.891]; open play worth ~5x) and dead-ball threat barely persists (r=+0.365 vs +0.815), but dead ball is only 21.8% of xG with a near-zero coefficient. |
| On-target xG faced (the save stage) | −0.0016 [−0.0086, +0.0047] | See below. |
| `deep_allowed`, `ppda` | +0.0002, −0.0021 | Defensive context adds nothing. |
| **`deep` completions (attack)** | **+0.0145 [−0.0172, +0.0442]** | **Best remaining candidate** — see BACKLOG. |

**Structural finding: `(count, total)` nearly spans the shot feed.** Every further decomposition is a sum over the same 116,448 shots and correlates ≥0.93 with shots or with total xG. A genuinely new channel must not be a function of the shot list — which is why `deep` (passes completed within ~20m of goal, not derived from shots) is the only live candidate left.

**On WP016:** on-target xG is the *proper* measurement of its shot-stopping idea. 43% of the xG in WP016's signal came from shots the keeper never faced (only 57.5% of conceded xG is on target), and the clean version persists 2.5x better (r = +0.273 vs +0.111, difference +0.162 [+0.068, +0.249]). It still adds nothing predictively, so WP016's verdict stands for a better reason than originally given, and the defensive-channel direction is closed.

## There is no such thing as a one-sided channel here

Worth recording because it looks like a gap and isn't. `xGA` is **identically** the opponent's `xG` (verified: max difference 0.0 across all paired rows). Because `theta` is additive in `attack − defence` and each match contributes two observations, `xG_obs_away` is scored against `lambda_away = exp(… + attack[away] − defence[home])` and therefore already constrains the **home defence** on every match. A separate "xGA channel" would be the same density, renamed — it would double-count. The same holds for this shot channel: `shots_away` is "shots conceded by home" as much as "shots created by away".

So channels are not added for attack vs defence. They are added for **distinct measurements of the scoring rate**, of which there are now three: goals, xG, and shot count.

## Cautions this WP added to the BACKLOG

7. **An in-sample R² gain from a nested model is not evidence** — adding a regressor can never reduce in-sample R², so the bootstrap lower bound is ≥0 by construction. Three premise checks here produced lower bounds of exactly +0.0000 that way. Coefficient-difference CIs are unaffected and stay valid.
8. **The 240-team-season premise check cannot certify anything** — OOS CI half-widths are ±0.03 R², wider than every effect of interest including the xG/shots premise itself. Use it to rank and to kill redundant candidates; let the CV harness decide.
9. **Match the premise check's aggregation level to what the model actually does.** Per-match (n=7,680) and half-season (n=240) rank shot count differently, and per-match is the one that matters. This is the single most useful thing WP019 produced.

## Reproducing

```bash
python work_products/wp019_shot_channel/prepare_data.py          # joins shot counts from WP014's cache
python work_products/wp019_shot_channel/run_cv.py --workers 4    # 90 tasks, ~141 min
python work_products/wp019_shot_channel/analyse.py
```

Needs WP007's `cv_shared_data.pkl`/`checkpoints/`, WP013's `checkpoints/`, WP008's `odds_raw.pkl`, WP014's `shots/`.
Also cached here for future work: `team_context.pkl` (9,120 team-matches of `deep`/`ppda`, 12 seasons) and `short_to_title.pkl` (Understat code → title join key).
