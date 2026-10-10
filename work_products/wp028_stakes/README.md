# WP028 — End-of-season stakes

**Status: complete. One arm ADDED to the champion, one measurably worse.**

| arm | ΔRPS vs champion | verdict |
|---|---|---|
| **`I_effort`** | **−0.00014 [−0.00031, +0.00001]** | **added** — best stability in the project |
| `J_disp` | **+0.00019 [+0.00008, +0.00031]** | **worse**, positive in 10/10 folds |

## Why this idea was different from the ones that failed

Incentive exists only **now**. A mid-table side in May has nothing to play for, and the latent walk learns "they played badly in April" only *after* those matches — it is forward-looking in a way the walk structurally cannot anticipate. And it needs **no new data**: the league table is already in `df_cv`.

## Measuring stakes

Stakes = how much **this** result moves the team's end-of-season probabilities. For each (season, round) the remaining fixtures are simulated 3,000 times; the team's current result is conditioned on win/draw/loss and P(outcome) recomputed. Stakes is the spread, summed over relegation / Europe / title.

That derives the four situations (title race, Europe, relegation, nothing to play for) from the arithmetic rather than imposing hand-drawn categories, and it separates "five points clear with three to play" from "level on points". **23.9% of matches are dead rubbers.**

**Boundaries: three, deliberately not more.** P(top 4) and P(top 5) sensitivities are near-collinear for most teams in most weeks, and WP011's crux test looked significant purely because correlated features split into cancelling coefficients. Relegation (bottom 3) is unambiguous every season; Europe uses the season's observed qualifier count (7–8 from 2020-21, assumed 6 before, since openfootball carries no Europa League data pre-2020); title is 1st.

## Premise

Residual correlation, n=3,780, team-clustered:

| | r |
|---|---|
| total stakes differential | **+0.0403 [+0.0111, +0.0673]** |
| April/May only (n=773) | **+0.0629 [+0.0161, +0.1111]** |
| each boundary alone | null |
| dead-rubber variance ratio | **1.160** (dead rubbers MORE variable) |

## Both arms avoid an additive covariate — on purpose

Congestion, absence, lineups and keeper all fitted at **~2% of a prior SD**, because an additive term competes with latents that already explain the variance. Both arms here were built to sidestep that, and both coefficients came back large (91% and 60% of prior SD, sign-consistent 115/115) — the architecture change worked even where the result didn't.

**`I_effort`** — multiplicative on each side's own latent contribution:
```
theta = intercept + (1+g·stakes_h)·attack[h] − (1+g·stakes_a)·defence[a] + hfa
```
A dead rubber compresses both sides toward the intercept: an even, average-scoring game.

**`J_disp`** — `goals ~ NegBinomial(mu=lambda, alpha = alpha0·exp(b·stakes_match))`. Touches the *variance*, so it cannot compete with the latents at all. **`predict.py`'s `scoreline_grid` takes a matching alpha** — an overdispersed likelihood with a Poisson predictive would measure nothing, since flatter probabilities in dead rubbers *is* the mechanism. Verified to reproduce Poisson to 1e-6 as `alpha → ∞` and to shrink the H−A spread monotonically as alpha falls.

## `J_disp` was my predicted winner and it is worse

I put it on record as the better bet: it was the mechanism the premise supported and the one that could not fight the latents. It is **reproducibly worse** — +0.00019, CI excluding zero, positive in **10/10** folds, better in only 1/10 seasons.

The dead-rubber variance observation (ratio 1.160) was real as a *conditional* fact and wrong as a *global model*. The arm imposes a smooth monotone relationship between stakes and dispersion on every match, so the 76% that are not dead rubbers pay a cost for the 24% that are.

## Why `I_effort` was added despite a CI including zero

It has the **best stability measured in this project**:

| | |
|---|---|
| leave-one-season-out | **negative 10/10 folds**, range −0.00008 to −0.00018 |
| per-season | better in 7/10 |
| coefficient | −0.272, **91% of prior SD**, sign-consistent **115/115** |
| external data | **none** — simulated from the league table in `df_cv` |

For contrast: `deep` halved on dropping 2016 (−0.00083 → −0.00044), and the Elo anchor passed 10/10 but needs a third-party mirror of a proxy-blocked API.

**The confound check cleared it.** The coefficient says a team at maximum stakes expresses roughly half its latent advantage — a large behavioural claim — and high stakes plausibly proxies for *mid-strength* (title races involve two clubs; by April the top sides are often already safe, while relegation and European races are fought by middling teams). That would make the term a regulariser on mid-table sides rather than a measure of incentive. It isn't:

| | r |
|---|---|
| raw | +0.0403 [+0.0130, +0.0678] |
| residualised on calendar position | +0.0403 |
| residualised on the model's strength gap | +0.0408 |
| **residualised on both** | **+0.0404 [+0.0114, +0.0678]** |

`corr(stakes diff, strength gap) = −0.089`, `corr(stakes diff, calendar) = −0.004`. Stakes is genuinely orthogonal to both.

**Honest caveat:** the effect is 5% of the xG channel and 4% of the gap to the opening line, and it shrank as the grid filled (−0.00026 at 56 windows → −0.00014 at 115). It is included on expected value with unusually good robustness, not because it is large.

## Consequence for the champion

`ModelConfig.champion()` now **requires** `ModelData.stakes_home/away`. A plain `ModelData` without them raises. Build with `stakes.py` then `prepare_data.py`.

## Three of my own bugs caught before they could manufacture findings

1. **`(cum_round, team)` is not unique** — ISO-week buckets let a team play twice, giving 1,170 duplicate keys. WP009 hit this exact trap (a join went 1205 → 2285 rows); the fix is to key on **(team, date)**.
2. **The round-date join** silently kept only matches on the round's *first* date — 38% of the sample, and a biased one, since TV picks are the bigger clubs. On that subset `stakes_title` looked "REAL" at +0.055 and April/May at +0.216; both fell apart on the full sample.
3. **Round-1 league scoring rate of 0.3** instead of ~1.35, from `max(observed, 0.3)` with nothing played. Collapsed simulated scorelines toward 0-0.

Also recorded: my intuition that early-season stakes were too high was **wrong**. Mean stakes fall late because fates get sealed (32% → 79% at ~zero), while the maxima rise. Three points is 0.39 SD of remaining points with 33 games left and 1.58 SD with 2.

## Reproducing

```bash
python work_products/wp028_stakes/stakes.py        # module; compute() called below
python work_products/wp028_stakes/prepare_data.py
python work_products/wp028_stakes/run_cv.py --workers 4   # 230 tasks, ~136 min
python work_products/wp028_stakes/analyse.py
```
