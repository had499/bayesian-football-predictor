# WP029 — Does the multiplicative form rescue congestion?

**Status: complete. Verdict: NO.** Congestion in the multiplicative effort term is worth **+0.00003 [−0.00001, +0.00008]** (rest) and **+0.00000 [−0.00005, +0.00005]** (load). The champion is unchanged.

## The hypothesis was architectural, not about congestion

Four **additive** covariates all fitted at ~2% of a prior SD and died: congestion (WP027), absence (WP012), lineups (WP011), keeper. WP028 then showed **non-additive** forms get used — its multiplicative effort term fitted at 91% of its prior SD, its dispersion term at 60%, WP025's Elo latent-prior at 169%. So: were those nulls about the **variables** or the **form**?

"A tired team plays below itself" is also a better mechanism than "a tired team scores 0.05 fewer goals", making congestion the natural first retest.

## Design

```
effort = 1 + g_stakes·stakes + g_cong·rest        (contributions sum INSIDE one multiplier)
theta  = intercept + effort_h·attack[h] − effort_a·defence[a] + hfa
```

The champion already carries a stakes effort term, so congestion joins the **same** multiplier rather than creating a second. `corr(stakes, rest) = +0.049`, so they stay separately identifiable. `rest` and `load` are separate arms because they correlate **−0.689** and WP011's crux test looked significant purely because correlated features split into cancelling coefficients.

Baseline is WP028's `I_effort`, which **is** the current champion.

## Results

| | complete calendar 2018-2024 (n=2,660) | full grid (n=3,780) |
|---|---|---|
| `K_rest_eff` | **+0.00003 [−0.00001, +0.00008]** | +0.00004 [−0.00001, +0.00009] |
| `L_load_eff` | **+0.00000 [−0.00005, +0.00005]** | +0.00002 [−0.00003, +0.00006] |

Headline is restricted to 2018-2024 because openfootball carries domestic cups only from 2018-19 and the Europa League only from 2020-21; for Understat seasons 2016/2017/2025 (25% of the grid) rest is league + Champions League alone, averaging 7.27 days against 6.38 for complete seasons. Both slices agree, so the mismeasurement wasn't hiding anything — but it's the restriction WP027 applied for the same reason, and `analyse.py` now does it by default rather than by memory.

## The coefficient DECAY is the real finding

| | first 30 windows | last 30 windows |
|---|---|---|
| `rest_effort` | +0.0326 (10.9% of prior SD) | **+0.0081 (2.7%)** |
| `load_effort` | +0.0051 | −0.0001 |

At 12 windows `rest_effort` sat at **17.6%** of its prior SD and looked like the architectural story was working. It decays monotonically to ~2.7% — i.e. back to where the *additive* version sat.

The mechanism is clear: early windows have the least training history, so the latents are weakest and an auxiliary covariate has the most room to explain residual variance. Given full history the model stops using it. **A promising coefficient on early CV windows is not evidence** — the whole point of 115 windows is that the late ones have realistic training depth.

Three partial reads (34, 48, 50 windows) also showed `K_rest_eff` "worse" with CIs excluding zero. All resolved to a precise zero at 115.

## Why multiplicative worked for stakes and not congestion

Not because multiplicative is magic — because the **mechanism must be quality-proportional**. The multiplicative form says *"you express less of the quality you have"*, which only means something if you have quality to lose. Measured for stakes:

| | r (stakes diff vs residual margin) |
|---|---|
| even games | +0.0105 [−0.0387, +0.0662] |
| middle | +0.0399 [−0.0059, +0.0865] |
| **mismatched games** | **+0.0651 [+0.0206, +0.1091]** |

Monotone, with only the mismatched tercile clearing zero. And the RPS gain from the stakes term sits entirely where the **favourite** has high stakes (−0.00035, CI excluding zero) versus +0.00020 where the favourite has low stakes. Rest shows no such proportionality, because it has no effect to scale.

**Note the direction**: the stakes coefficient is **negative** (−0.272), so high stakes *suppress* expressed quality — pressure, not dead-rubber complacency. A strong favourite in a title decider plays nearer to average.

## Bottom line

Congestion is now closed from four independent directions: additive null with the tightest intervals in the project (WP027), a residual-correlation premise that was null *and the wrong sign* for fatigue, a multiplicative coefficient that decays to the additive level as training history accumulates, and RPS of +0.00003.

And the architectural lesson is properly bounded: **non-additive forms help only where the effect genuinely scales with team quality.** That lowers expectations for retesting absence or keeper the same way — WP011/WP012 found no main effect for those to scale in the first place.

## Reproducing

```bash
python work_products/wp029_congestion_effort/prepare_data.py
python work_products/wp029_congestion_effort/run_cv.py --workers 4   # 230 tasks, ~116 min
python work_products/wp029_congestion_effort/analyse.py
```
