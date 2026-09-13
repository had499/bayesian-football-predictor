# WP003 — Bookmaker Benchmark

**Status: complete. Conclusion — the model beats trivial baselines but is decisively not competitive with the market, and has no exploitable subset.**

## TL;DR

- Pooled RPS over 401 held-out matches: **naive 0.234 → model 0.196 → Bet365 0.183 → Pinnacle 0.179**. The model closes ~68% of the gap between "know nothing" and the sharp closing line.
- Model vs. Pinnacle, paired: **+0.0139 RPS, 95% CI [+0.0080, +0.0198]** — reliably ~8% worse than the closing line, CI well clear of zero, unchanged under Shin de-vig.
- **No edge, anywhere.** The more the model disagrees with the market, the *more wrong* it is (gap grows monotonically: +0.014 overall → +0.041 on the top-10% disagreements). Promoted-team matches early season — the one place bookmakers are supposedly soft — are where the model's gap to the market is *largest*. A toy betting sim loses money at every threshold and loses more as the required edge increases.
- **Mechanism**: calibration vs. Pinnacle shows the model over-shrinks — under-confident on home favourites (predicts 0.78 where the true rate is 0.93), slightly over-confident on underdogs. Pinnacle is near-perfectly calibrated. This is the same calibration gap flagged in WP001, now confirmed against a real reference.
- **Verdict**: as a forecaster, ~2/3 of the way to market quality; as a betting tool, worthless as-is. The concrete next lever is the over-shrinkage.

WP001 established the model beats a naive baseline (league-average goals) on RPS, statistically significantly. WP002 established that neither xG nor Dixon-Coles contributes to that — the signal is entirely in the base hierarchical attack/defense/home_adv/AR(1) model plus six seasons of data. Both results were measured against a bar so low that "beats it" doesn't tell you whether the model is any *good*.

This work product measures the model against the real bar: **the bookmaker market**. Not framed as "can we beat the closing line" — we almost certainly can't, the market has team news and late money the model doesn't — but as two questions with genuinely unknown answers:

1. **How large is the gap to the market, and where is it?** A denominator for every future model change. "Closed 15% of the gap to Pinnacle" is interpretable; "improved RPS 0.002 over league average" is noise.
2. **Are there subsets where the model has a real edge over the market?** A model worse *on average* can still be right where it disagrees most, on match types bookmakers are known to misprice (promoted teams early season, post-managerial-change, dead rubbers). If such a pocket exists, that subset is the actual product. If it doesn't, that's a finding too — stop polishing.

## What this is

No retraining. This reuses WP001's 35-window walk-forward CV output verbatim — the same 401 held-out match predictions (`lambda_home`/`lambda_away` per match, converted to H/D/A probabilities via `dc_outcome_probs`) — and adds bookmaker odds for those exact fixtures as an additional "predictor" in the comparison harness WP001/WP002 already use (pooled RPS, paired bootstrap, calibration tables).

"The model" here is WP001's `cv_checkpoint.pkl` (`use_xG=True, use_dixon_coles=True`). Per WP002 the feature flags don't matter, so this equally represents the simpler `neither` config.

## Method

### Odds data — historical archive, not live scraping

The 401 CV test matches were played across EPL 2020-21 → 2025-26. You cannot scrape odds for matches that already happened — live bookmaker sites only list upcoming fixtures, and accumulating odds "over time" from today forward would never cover a backward-looking CV. The right source is a historical archive:

**[football-data.co.uk](https://www.football-data.co.uk/englandm.php)** — the canonical free source. One CSV per season (`mmz4281/<code>/E0.csv`), each with closing 1X2 odds from multiple books:

- **Pinnacle** (`PSCH`/`PSCD`/`PSCA`) — the sharp benchmark. Low margin (~2%), high limits, so its closing line is the most efficient public estimate of true probability. This is the number that matters.
- **Bet365** (`B365CH`/`B365CD`/`B365CA`) — a mainstream recreational book (~5-7% margin), included to show the spread between sharp and soft.
- **Market max / average** (`MaxCH…`, `AvgCH…`) — best-available price and cross-book average across ~5-10 books. `Max` is what you'd actually bet into; `Avg` is the consensus.

If a season's file lacks the closing (`*C*`) columns, fall back to the pre-closing (`PSH`, `B365H`, …) columns and note it.

### Join

Model predictions carry team codes (`ARS`, `MCI`, …) and a datetime; football-data.co.uk uses its own names and a date. Join on `(match date, home team, away team)` via a fixed code→name crosswalk (in the notebook; only six names differ from the obvious — `Man City`, `Man United`, `Newcastle`, `Nott'm Forest`, `West Brom`, `Wolves`). The notebook prints any unmatched fixtures and asserts every model prediction's stored `(goals_home, goals_away)` equals the joined row's actual score — a hard integrity check that the two datasets are describing the same match.

### Odds → probabilities (de-vigging)

Raw implied probabilities `1/odds` sum to >1 (the bookmaker's margin / "overround"). Default method: **proportional** — divide each by the sum so they total 1. Simple, and for Pinnacle (~2% margin) the method barely matters. For higher-margin books, proportional slightly overstates favourites; **Shin's method** is the standard refinement (accounts for the favourite-longshot bias / informed-bettor effect) and is included as an alternative to check sensitivity.

### Metrics

- **Pooled RPS**, 401 matches: model vs. Pinnacle vs. Bet365 vs. market-average vs. WP001's naive baseline. Same `rps_home_draw_away` as WP001/WP002.
- **Paired bootstrap** of `RPS(model) − RPS(Pinnacle)` per match — the headline gap, with a 95% CI. A CI that excludes zero and sits positive = the model is reliably worse than the sharp line (expected); how *far* positive is the point.
- **Calibration**: model reliability table vs. the market's own reliability table on the same 401 matches. The market should be near-perfectly calibrated by construction; where the model's curve departs from it is where the model is losing.

### Edge detection (the part with an unknown answer)

- **Disagreement buckets**: sort matches by `|P_model(home) − P_market(home)|`. On the top decile/quartile — where the model most disagrees with the market — is the model's RPS on those matches *better* or *worse* than the market's? Better on the tail = a real edge worth isolating.
- **Promoted-team subset**: teams appearing in a season but not the one before (derivable from the data). Bookmakers are historically soft on these early in the season. Does the model beat the market on matches involving a newly-promoted side in the first ~10 rounds?
- **Toy betting sim (illustrative only)**: flat-stake bet on any outcome where `P_model` exceeds the `Max` (best-available) implied probability by a threshold τ; report ROI and its bootstrap CI over a τ sweep. This is in-sample to the τ choice, so it's a diagnostic ("is there anything here at all"), not a strategy.

### Caveats baked in

- Closing odds include information (team news, lineups, late money) the model never sees. This is a favourable bar for the market by design — "within X of the closing line without knowing who's injured" is the honest way to read a small gap.
- Pooled RPS over 401 matches has limited power for small effects (WP002 saw this). A null on "model vs. market overall" won't be surprising or informative; the edge-detection subsets are where a real result would show up.
- The naive baseline in WP001's per-window `ll_naive` is training-only; the pooled naive RPS here is computed over the whole dataset for simplicity, same as WP001's pooled figure. Consistent with WP001, slightly generous to the baseline.

## Results

401 model predictions, all 401 joined to football-data.co.uk closing odds with zero score mismatches. Pinnacle closing odds (`PSC*`) are present for 361 of the 401; Bet365 / market-average / market-max cover all 401 and tell the same story, so the Pinnacle-subset figures below aren't an artefact of that gap.

Mean overround (bookmaker margin): Pinnacle 2.8%, market-average 4.3%, Bet365 5.5%. (Market-*max* is −0.5% — best price across all books can exceed a fair book; that's why it's used for the betting sim, not the RPS comparison.)

### Pooled RPS — model vs. market vs. naive

| Predictor | n | Pooled RPS | 95% CI |
|---|---|---|---|
| naive (league-average goal rate) | 401 | 0.2341 | [0.2289, 0.2391] |
| **model** | 401 | **0.1963** | [0.1851, 0.2081] |
| Bet365 (closing) | 401 | 0.1833 | [0.1714, 0.1958] |
| market average (closing) | 401 | 0.1834 | [0.1713, 0.1960] |
| Pinnacle (closing) | 361 | 0.1786 | [0.1659, 0.1919] |

The model sits between naive and the market, much closer to the market: it closes `(0.2341 − 0.1963) / (0.2341 − 0.1786) ≈ 68%` of the distance from "know nothing" to the sharp line. But it's clearly worse than every bookmaker, including the recreational one.

### Paired bootstrap — model minus Pinnacle, per match (n = 361)

| | Mean RPS diff | 95% CI |
|---|---|---|
| proportional de-vig | **+0.0139** | [+0.0080, +0.0198] |
| Shin de-vig | +0.0142 | [+0.0083, +0.0202] |

`+7.8%` relative to Pinnacle's own RPS. The CI excludes zero on both de-vig methods — the model is **reliably worse than the closing line**, not within noise of it.

### Calibration — model vs. Pinnacle, P(home win)

| Bin | Model n | Model pred | Model actual | Pinnacle n | Pin. pred | Pin. actual |
|---|---|---|---|---|---|---|
| 0.00–0.12 | 7 | 0.10 | 0.00 | 23 | 0.09 | 0.04 |
| 0.12–0.25 | 64 | 0.20 | 0.17 | 59 | 0.20 | 0.20 |
| 0.25–0.38 | 96 | 0.32 | 0.35 | 73 | 0.31 | 0.33 |
| 0.38–0.50 | 114 | 0.44 | 0.47 | 79 | 0.44 | 0.43 |
| 0.50–0.62 | 65 | 0.56 | 0.52 | 62 | 0.56 | 0.53 |
| 0.62–0.75 | 40 | 0.67 | 0.75 | 42 | 0.69 | 0.88 |
| 0.75–0.88 | 14 | 0.78 | 0.93 | 22 | 0.81 | 0.82 |

Pinnacle is near-perfectly calibrated through 0.12–0.62 (every bin within ~2 points). The model **over-shrinks**: under-confident on home favourites (0.75–0.88 bin predicts 0.78, actual 0.93) and slightly over-confident on underdogs (its lowest bins over-predict home wins). Both model and Pinnacle wobble in the 0.62–0.75 bin, but that's small-n (n≈40) and the market recovers in the next bin while the model doesn't. This is the WP001 calibration gap, confirmed against a properly-calibrated reference.

### Edge detection — RPS gap to Pinnacle by disagreement decile

| Subset | n | Model RPS | Market RPS | Gap | 95% CI |
|---|---|---|---|---|---|
| all matches | 361 | 0.1925 | 0.1786 | +0.0139 | [+0.0080, +0.0198] |
| top 50% disagreement | 181 | 0.2030 | 0.1798 | +0.0232 | [+0.0124, +0.0339] |
| top 25% | 91 | 0.2045 | 0.1659 | +0.0386 | [+0.0203, +0.0556] |
| top 10% | 37 | 0.2208 | 0.1797 | +0.0411 | [+0.0069, +0.0744] |

**The gap grows monotonically with disagreement.** Where the model most departs from the market, it is *most* wrong — every CI positive. The model's disagreements with the line are error, not signal. There is no subset where it knows something the market doesn't.

### Promoted-team subset

Promoted sides per season, from the data: 2021-22 BRE/NOR/WAT, 2022-23 BOU/FLH/NOT, 2023-24 BUR/LUT/SHE, 2024-25 IPS/LEI/SOU, 2025-26 BUR/LED/SUN.

| Subset | n | Model RPS | Market RPS | Gap | 95% CI |
|---|---|---|---|---|---|
| promoted involved, rounds 1–10 | 29 | 0.1626 | 0.1349 | +0.0277 | [+0.0109, +0.0456] |
| promoted involved, round 11+ | 79 | 0.1660 | 0.1508 | +0.0152 | [+0.0037, +0.0257] |
| no promoted team | 253 | 0.2042 | 0.1923 | +0.0120 | [+0.0048, +0.0193] |

The hypothesis was "bookmakers are soft on promoted teams early — maybe the model has an edge there." The opposite: the model's gap to the market is *largest* on early-season promoted-team matches (+0.028, more than double the +0.012 on ordinary matches). Both model and market post low absolute RPS here — these fixtures are often lopsided and predictable — but the market's margin over the model is widest exactly where we hoped the model would be strong.

### Toy betting sim (illustrative, in-sample to the threshold)

Flat 1-unit stake on any outcome where `P_model` exceeds the best-available (`Max`) implied probability by threshold τ, settled at `Max`.

| τ | n bets | ROI | 95% CI |
|---|---|---|---|
| 0.00 | 582 | −16.9% | [−31.1%, −1.4%] |
| 0.02 | 429 | −23.0% | [−39.7%, −5.4%] |
| 0.05 | 264 | −24.4% | [−43.7%, −3.2%] |
| 0.08 | 126 | −27.6% | [−56.0%, +5.4%] |
| 0.10 | 82 | −29.4% | [−66.1%, +17.9%] |
| 0.15 | 19 | −52.1% | [−100.0%, +16.6%] |

Loses money at every threshold, and loses *more* as the required "edge" increases — the same signal as the disagreement deciles: the bets the model is most confident are value are the ones it's most wrong about. Wide CIs at high τ are just small n; the point estimates never turn positive.

## What this means for the project

1. **The model is legitimate but not market-competitive.** It's ~2/3 of the way from a trivial baseline to the sharp closing line, without using lineups, injuries, or news. That's a real result. It is also decisively behind every bookmaker, and that gap is statistically solid.
2. **There is no exploitable edge, and no obvious subset to carve out.** Every slice tested — largest disagreements, promoted teams early season, betting-sim thresholds — shows the model is *more* wrong exactly where it's most confident it disagrees with the market. Chasing a "the model is worse on average but has a profitable niche" story is not supported.
3. **The next lever is calibration / over-shrinkage**, not features (WP002 already ruled those out). The model pulls probabilities toward the middle — it needs to be more willing to back strong favourites. Candidate directions: weaker shrinkage priors on `attack`/`defence` and `home_adv`, per-team `sigma` (partial pooling) so standout teams can separate from the pack, or an explicit recalibration layer (Platt / isotonic) fit on held-out CV folds. Whichever, WP003 is now the yardstick — a change is worth keeping only if it closes the gap to Pinnacle, measured on this same 401-match comparison.

## Reproducing

```bash
cd work_products/wp003_bookmaker_benchmark
jupyter lab wp003_bookmaker_benchmark.ipynb
```

Run top to bottom. The odds fetch cell pulls six CSVs directly from football-data.co.uk (no auth, ~1 MB total) and caches them to `odds_raw.pkl` in this folder; delete that to re-fetch. Everything downstream is pandas — no PyMC, no sampling, runs in seconds. If football-data.co.uk is unreachable, download the season CSVs manually from the England page and point the fetch cell at local paths.
