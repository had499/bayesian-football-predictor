# WP012 — Market inefficiency: is there any bettable edge?

**Status: complete. Result: no primary test hit, and the model is not a source of edge. The design and decision rules below were written before the notebook was run; it was run once, with no test or threshold changed afterwards. See "Results". Update: WP015 repeated P1 and P2 on 998 fresh matches and reached the same conclusion, but the −23% realised ROI below did not replicate (−3.5%, CI [−14.6%, +7.5%] on fresh matches); do not quote it.**

## Why

The project's goal is to beat bookmaker odds. WP003 measured the model against Pinnacle's closing line and found it ~8% worse in RPS (+0.0139, CI [+0.0080, +0.0198]), with a flat-stake 1X2 betting sim that lost at every threshold. WP005/006/009/011 then tried four different ways to improve the model and none moved the gap; WP010 concluded to stop chasing accuracy on this architecture and to re-examine "no edge anywhere" beyond the one betting rule WP003 tried.

This WP asks a different question from every WP since WP003: not "how do we make the model better" but **"is there any exploitable inefficiency in this odds data at all, and if so does the model have anything to do with it?"** No retraining. It is a notebook over data already in the repo.

## Data

`work_products/wp003_bookmaker_benchmark/odds_raw.pkl`: football-data.co.uk EPL odds, 2,280 matches (6 seasons), full-time results, and for Pinnacle / Bet365 / market-average / market-best both a pre-closing price (`PS*`, `B365*`, `Avg*`, `Max*`) and the closing price (`PSC*`, `B365C*`, `AvgC*`, `MaxC*`). Pinnacle coverage is 92.5% of matches; the other three books are complete. The model's held-out predictions (WP001 baseline checkpoint, 401 matches, 361 with Pinnacle odds) are joined to it exactly as WP003/WP011 did, now with a hard check that every joined score matches the odds file.

Assumptions worth naming: the "Max" price is the best across a set of bookmakers that may include Pinnacle itself (unverified); pre-closing prices are treated as "available earlier in the week" without knowing the exact snapshot time; closing prices of different books are treated as near-simultaneous.

## What was built

- **`src/football_model/evaluation/market.py`** — small pure-numpy helpers: de-vig, RPS, match-clustered bootstrap for ROI and CLV, bet settlement, value-bet ROI tables, odds-band tables, calibration tables, leave-one-window-out multinomial-logit log loss, and the fixture/odds join.
- **`tests/test_market.py`** — 23 tests. Beyond hand-computed cases, three simulate a market with a *known* edge: the pipeline must recover a +5% edge inside its CI, recover a −5% margin and place no bets against it, and — the important one — a bettor whose beliefs are independent of the truth, betting against a fair book, must get an ROI CI containing zero. Bet settlement and the bootstrap are where a bug silently manufactures an edge instead of crashing.
- **`wp012_market_inefficiency.ipynb`** — the analysis.

## Analysis

1. **Market-only, all 2,280 matches (model-free).** Pinnacle closing calibration (favourite-longshot, draw and home bias); blind ROI by odds band for each book; **soft-vs-sharp value bets** — bet a soft book's closing price whenever Pinnacle's de-vigged closing probability says it is worth more than `tau`.
2. **Does the model add information beyond Pinnacle? (361 matches)** Blend sweep `(1−w)·Pinnacle + w·model` with paired RPS bootstrap, and leave-one-window-out multinomial logits: market log-odds only vs market plus model log-odds.
3. **Opening vs closing.** Model and Pinnacle-open RPS against the close; whether the model's disagreement with the opening line predicts line movement; and the **closing line value (CLV)** of bets the model would place at Pinnacle's opening odds.

## Pre-declared primary tests

Three, to limit the false-positive rate (at most about 7% that at least one passes by chance if there is truly nothing). Everything else in the notebook is exploratory and does not count as a finding.

| # | test | a "hit" requires |
|---|---|---|
| P1 | leave-one-window-out log loss, market+model minus market-only (per match, 361 matches) | 95% CI entirely below zero **and** same sign in both halves of the data |
| P2 | mean CLV per bet of model-selected bets at Pinnacle opening odds, `tau = 0.02`, vs Pinnacle's closing fair prices | 95% CI entirely above zero **and** same sign in both halves |
| P3 | ROI of flat-staking Bet365 closing odds where Pinnacle's closing fair prices give an edge above `tau = 0.02` | 95% CI entirely above zero **and** same sign in both halves |

"Halves" = split at the median match date. `tau` was fixed here, not tuned; the `tau` sweep in the notebook is diagnostic.

## What each outcome would mean

- **No hits.** No bettable edge from this model, and none from soft-book-vs-Pinnacle gaps in EPL closing odds, at this power. Stop the "beat the bookies with this model" line; what remains is new information (not more of the same historical data) or different markets (over/under, Asian handicap).
- **P3 only.** The exploitable thing is not the model but line shopping against a sharp reference. Build around Pinnacle as the reference price.
- **P1 and/or P2.** The model carries information the market lacks. Required next step is a forward test on live odds it has never seen before any real staking, since a historical hit from three tests is a lead, not proof.

## Results

**No primary test hit.**

| # | test | n | estimate | 95% CI | half 1 / half 2 | result |
|---|---|---|---|---|---|---|
| P1 | log loss, market+model − market-only (negative = model helps) | 361 | +0.0070 | [−0.0040, +0.0183] | +0.0075 / +0.0065 | no hit |
| P2 | CLV per bet at Pinnacle opening odds, `tau=0.02` | 416 bets | −3.0% | [−4.1%, −1.9%] | −1.9% / −4.2% | no hit |
| P3 | Bet365 closing ROI vs Pinnacle fair, `tau=0.02` | 75 bets | −22.9% | [−69.7%, +34.3%] | +41.1% / −100% | no hit, **uninformative** (see below) |

**The model adds nothing to the market, and by every measure tried it makes it slightly worse.**
- Blending any weight of the model into Pinnacle's close raises RPS, monotonically: +0.0004 at `w=0.05` (CI [+0.0001, +0.0007], already significant), +0.0019 at 0.2, +0.0055 at 0.5, +0.0139 at 1.0 (pure model, which reproduces WP003's number).
- P1: adding the model's log-odds to a recalibrated Pinnacle worsens leave-one-window-out log loss by 0.007 per match, with both halves the same sign.
- The model is not a better opening line either: RPS 0.1925 vs Pinnacle's opening 0.1808 (model − open +0.0117, CI [+0.0063, +0.0172]). The market does get more accurate towards the close (open − close +0.0023, CI [+0.0001, +0.0044]), which is the expected direction and a sanity check that the opening and closing columns behave as labelled.
- Where the model disagrees with the opening line has no relationship to where the line then moves: slope −0.013, CI [−0.049, +0.024].
- P2: the bets the model likes get a price no better than random ones. CLV is −3.0% at `tau=0.02` against −2.9% for betting every outcome (the control, about Pinnacle's margin), and gets worse as the threshold rises (−3.4% at 0.05). Their realised ROI is −23% (CI [−39%, −7%]), matching WP003's betting sim.

**The market itself (2,110–2,280 matches, exploratory):**
- Pinnacle's closing line is well calibrated: no probability bin's hit rate differs significantly from its mean probability, and neither does home/draw/away frequency.
- Blind ROI at closing odds, every outcome: Pinnacle −3.9% (CI [−6.3%, −1.4%]), Bet365 −6.5%, market-average −5.4%, **best available price ≈ 0 (+0.03%, CI [−2.4%, +2.4%])**. So shopping for the best price recovers the whole margin but earns nothing on top; any profit would have to come from skill, and this model has none relative to the market. (Caveat from the Data section: the best-price series may include Pinnacle and stale or unavailable quotes, so this is an upper bound.)
- Favourite-longshot: long-odds bands lose more (Pinnacle −7.1% at odds 5–8, −11.7% at 8+; Bet365 −26.6% at 8+), but the CIs of every band at odds 5 and above are very wide and all include zero (the shorter bands' negative CIs are just the margin). Suggestive only.

**P3 was underpowered and should not be read as "no soft-book edge".** Bet365's closing price beat Pinnacle's fair price by more than 2% on only 75 of 6,330 outcomes, and those bets hit only 11% of the time, so they are mostly long shots and very high variance, so the ROI CI spans −70% to +34%. Best-price versus Pinnacle fair had 1,971 qualifying bets with a claimed edge of +6.2% and a realised ROI of −0.9% (CI [−9.8%, +7.7%]): price gaps that Pinnacle's fair prices call worth 6% were worth about nothing, and this data cannot say whether the gaps are unavailable outlier quotes or Pinnacle being wrong.

**Decision (by the rules above).** No hits: stop pursuing "beat the bookies with this model" on EPL 1X2. That is a stronger statement than WP003's, because it now covers the opening line, CLV, and residual information, not just closing RPS. Two questions remain open, both about the market rather than the model:
1. **Soft-vs-sharp at higher power.** Needs more matches than 2,280 EPL games gives; football-data.co.uk also publishes files for the other top leagues, which would add roughly 8,000 matches if they carry the same Pinnacle and closing columns (not yet checked), and the test should be on a lower-variance measure than realised ROI.
2. **Other markets.** Over/under 2.5 and Asian handicap odds are already in `odds_raw.pkl` and the model's score distribution prices them directly. Untested; a null is likely, but it has not been checked.

## What this cannot show

- Realised profit. Bets are settled at published odds with no stake limits, no account restrictions, no liquidity and no commission. Soft bookmakers restrict winning accounts; a positive P3 is an upper bound on what is realisable.
- Anything about markets not in the data (over/under, Asian handicap) — that is the follow-up.
- P1 and P2 have n ≈ 361, so only large effects are detectable; a null there means "not large", not "zero".

## Reproducing

```bash
pytest tests/test_market.py
cd work_products/wp012_market_inefficiency
jupyter lab wp012_market_inefficiency.ipynb   # ~ a few minutes; no sampling
```
