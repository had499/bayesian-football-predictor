# WP015 — Re-testing "the model adds nothing to the market" on fresh matches

**Status: complete. Neither pre-declared primary test hit on 998 fresh Pinnacle-covered matches, so WP012's conclusion holds on independent data: the model adds nothing to Pinnacle's price. One WP012 number did not replicate (the realised ROI of model-picked bets). See "Results".**

## Why

WP012 found the model adds nothing to Pinnacle's price (blending it in raised RPS; adding it to a recalibrated Pinnacle worsened log loss; bets it liked got no better price than random). That used the same 361 Pinnacle-covered matches as WP001–013, so it rests on one sample. WP014 produced **998 Pinnacle-covered matches that were never held out before** (1,086 new matches in total), with all four model arms fitted on them. This WP repeats WP012's two model-dependent tests on those matches.

WP012's third test (soft book vs Pinnacle, P3) does not use the model, so there is nothing to re-test.

## Design

- **Model predictions:** WP014's checkpoints (`work_products/wp014_continuity_confirmation/cv_checkpoint_<arm>.pkl`), test rounds disjoint from all of WP001's. Every joined score is checked against the odds file.
- **Primary model = `baseline`**, the same model WP012 tested, so this is a like-for-like replication. `continuity_lineup_loose_combo` (the best arm) is reported as exploratory.
- **Same test functions as WP012** (`football_model.evaluation.market`), same 95% bootstrap and the same halves rule (split at the median match date).
- **Caveat that biases against the model:** WP014's predictions are up to 3 rounds ahead instead of 1, which costs accuracy. A miss here is therefore slightly easier than in WP012.

## Pre-declared primary tests

| # | test | a "hit" requires |
|---|---|---|
| P1 | leave-one-window-out log loss, market + model minus market only (per match); negative = model helps | 95% CI entirely below zero and the same sign in both halves |
| P2 | mean CLV per bet of model-selected bets at Pinnacle's opening odds, `tau = 0.02`, against Pinnacle's closing fair prices | 95% CI entirely above zero and the same sign in both halves |

Both on `baseline`, on the fresh matches only. Everything else (blend sweep, opening-vs-closing RPS, line-movement slope, the best arm, all other `tau` values) is exploratory.

## What each outcome means

- **Neither hits.** WP012's conclusion holds on independent data: this model adds nothing to Pinnacle's price on EPL 1X2. Stop pursuing "beat the bookmakers with this model" here.
- **P1 or P2 hits.** The model carries information the market lacks, contradicting WP012. That would be a lead, not proof: it needs a forward test on odds the model has never seen before any real staking.

## Power

WP012 measured P1 at n = 361 with CI [−0.0040, +0.0183] (half-width about 0.011); at 998 matches the half-width should be about 0.007. So P1 can detect a log-loss gain of roughly 0.007 per match or larger. That is a coarse instrument, and a miss rules out a large gain, not a small one.

## Results

998 fresh matches with Pinnacle pre-closing and closing odds (of 1,086), model = `baseline`. Pinnacle closing RPS 0.1982, Pinnacle opening 0.1993, model 0.2056. The notebook was run once.

| # | test | n | estimate | 95% CI | half 1 / half 2 | result |
|---|---|---|---|---|---|---|
| P1 | log loss, market+model − market-only (negative = model helps) | 998 | +0.0017 | [−0.0030, +0.0064] | +0.0007 / +0.0028 | **no hit** |
| P2 | CLV per bet at Pinnacle's opening odds, `tau=0.02` | 1,115 bets | −3.2% | [−3.9%, −2.5%] | −2.4% / −4.1% | **no hit** |

(WP012 on the original 361: P1 +0.0070 [−0.0040, +0.0183]; P2 −3.0% [−4.1%, −1.9%].) The estimates agree closely, and both are on the unfavourable side.

**What replicated**
- **The model does not help.** Every blend of model into Pinnacle's close has a positive (worse) point estimate: +0.0001 at `w=0.05`, +0.0005 at 0.2, +0.0022 at 0.5, +0.0074 at 1.0. Only `w ≥ 0.5` is significantly worse; on WP012's sample even 5% was.
- **Model-selected bets get no better a price than random ones.** CLV −3.2% at `tau=0.02` against −3.0% for betting every outcome (the control, about Pinnacle's margin), and it does not improve as the threshold rises (−3.4% at 0.05).
- **The model is not a better opening line**: model − Pinnacle open = +0.0063 RPS, CI [+0.0027, +0.0098].
- The best arm (`continuity_lineup_loose_combo`, exploratory) is closer to Pinnacle (RPS 0.2044; gap +0.0063) but still does not help: P1 +0.0007 [−0.0048, +0.0060], P2 CLV −3.0% [−3.7%, −2.3%], both no hit.

**What did not replicate: the realised ROI.** WP012 reported that the model's picks lost 23% (CI [−39%, −7%]). On fresh matches the same rule gives **−3.5%, CI [−14.6%, +7.5%]**, which includes zero. WP012's −23% was largely luck of the 416 bets; ROI has enormous variance, which is why CLV is the better measure and it did replicate. The 23% figure should not be quoted.

**Exploratory, one test among many: the market moves against the model.** The slope of (closing − opening) on (model − opening) is **−0.028, CI [−0.049, −0.005]** for the baseline (WP012: −0.013, CI [−0.049, +0.024]; best arm here −0.011, CI [−0.034, +0.011]). Negative means that when the model disagrees with the opening line, the line tends to move the opposite way: the model's disagreements look like noise the market corrects, not early information. It is one of about a dozen exploratory numbers in this notebook and is not significant for the best arm, so treat it as a hint.

**Verdict by the declared rule:** neither primary hit, so WP012's conclusion stands on independent data. The model, as built, has no bettable edge on EPL 1X2. This is consistent with everything since WP003.

**Limits, unchanged:** no realised-profit test (no limits, restrictions, liquidity); EPL 1X2 only, 2020–2026; WP014's 3-round-ahead predictions handicap the model slightly. P1's half-width is about 0.005, so it rules out a large log-loss gain, not a small one.

## What this cannot show

- Realised profit (no stake limits, restrictions or liquidity in the data).
- Anything beyond EPL 1X2, or beyond 2020–2026.

## Reproducing

```bash
cd work_products/wp015_market_retest
jupyter lab wp015_market_retest.ipynb   # no sampling, about a minute
```
