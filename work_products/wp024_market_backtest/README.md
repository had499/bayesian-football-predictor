# WP024 — Would betting this model make money?

**Status: complete. Verdict: NO — no evidence of profitability at any book, threshold, or staking scheme.** The best *tradeable* result is bet365 opening prices at a 2% edge filter: **+0.12% ROI [−6.51%, +6.65%]** over 3,424 bets. Every interval includes zero. Quarter-Kelly loses everywhere.

## Setup

The champion (`L0` + xG channel + deep channel) on WP021's 115-window grid — **3,767 walk-forward matches**, every prediction genuinely out-of-sample. Opening prices throughout, as asked: WP010 measured the model +0.0039 RPS behind Pinnacle's *pre-close* line versus +0.0139 behind the close, so openers are where it is closest to competitive.

Four price series, spanning optimistic to realistic:

| series | what it is | tradeable? |
|---|---|---|
| `market_max` | best price across all books | **NO — see below** |
| `bet365` | soft book | yes, until you get restricted |
| `pinnacle` | sharp book | yes, and sustainable — Pinnacle tolerates winners |
| `market_avg` | consensus | reference only |

**`market_max` is not a real price.** Its mean overround is **1.0049**, and **32% of matches show overround < 1.00 — pure arbitrage**. A simultaneously-available price set cannot contain a risk-free arb in a third of matches; this is a maximum taken across books *and across time*. Any return computed on it is an unachievable upper bound.

## Results — flat stakes, opening prices

| book | edge ≥2% | bets | ROI | CI (window-clustered) |
|---|---|---|---|---|
| `market_max` *(not tradeable)* | | 3,171 | +1.70% | [−4.58%, +7.90%] |
| `bet365` | | 3,424 | **+0.12%** | [−6.51%, +6.65%] |
| `pinnacle` | | 3,741 | −0.93% | [−6.69%, +4.98%] |
| `market_avg` | | 2,329 | −2.11% | [−9.13%, +4.99%] |

Raising the threshold to 5% or 10% does not help. Quarter-Kelly is **worse than flat at every book** (−2.78% to −9.29%).

## Why it fails — the model is accurate but not *differently* accurate

**Calibration is excellent.** Predicted vs actual frequency agrees within 0.03 in every bucket:

| model p | bets | predicted | actual |
|---|---|---|---|
| 0.2–0.3 | 4,282 | 0.248 | 0.250 |
| 0.4–0.5 | 1,275 | 0.448 | 0.439 |
| 0.6–0.7 | 594 | 0.645 | 0.675 |
| 0.7–1.0 | 519 | 0.771 | 0.780 |

So this is not a calibration failure — consistent with WP004/WP009, where post-hoc recalibration netted to zero.

**But the claimed edges are mostly noise**, and destructively so at the extremes:

| claimed edge | bets | ROI |
|---|---|---|
| ≤ 0% | 7,407 | −6% to −10% |
| 0–5% | 1,060 | −0.15% |
| 5–10% | 791 | +4.34% |
| 10–20% | 946 | +5.50% |
| 20–40% | 714 | −6.03% |
| **>40%** | 383 | **−14.47%** |

The model loses *most* where it is *most confident* it has an edge. That is exactly why Kelly underperforms flat staking: Kelly sizes up on the bets that are worst.

**The mid-band is not a strategy.** Betting only 5–20% claimed edge gives +4.97% in-sample on bet365 — but that band was chosen by reading the table above. Split by season, selecting on 2016–20 and holding out 2021–25:

| book | selection | held out |
|---|---|---|
| bet365 | +7.05% | +2.62% |
| market_max | −6.45% | **+6.87%** |
| pinnacle | +7.08% | **−6.63%** |

Two of three flip sign. Noise.

## The structural reason

A bettor needs to be **more accurate than the price**, not merely accurate. This model is **+0.0044 RPS behind Pinnacle's opening line**. Being well-calibrated *and slightly worse than the market* means apparent "value" is mostly the model's own error, which is precisely what the edge-bucket table shows.

Bookmaker margins compound it: ~4.6% overround at bet365 openers, ~2.7% at Pinnacle. The model would have to beat the line by more than the margin, consistently, at prices it can actually get.

## What this does NOT model — all of which make real returns worse

* **Stake limits at opening prices**, which are typically very low
* **Account restriction** at soft books, the standard outcome of persistent winning
* **Price movement** between the recorded open and when a bet could be placed
* **`market_max` being unobtainable**, as above

## Bottom line

The honest read is that this is what a well-calibrated model that is slightly behind the market looks like: no edge, no ruin, returns indistinguishable from zero minus the vig. Nothing here suggests a tradeable strategy, and the one pattern that looked promising did not survive a season split.

To revisit this, the model would need to *beat* the opening line on RPS rather than trail it by 0.0044.

## Trading only on DISAGREEMENT — also no, and it is worse

Backing the outcome where the model is most bullish versus the de-vigged Pinnacle
opener, restricted to the most-disagreeing matches:

| disagreement cut | ROI backing the model |
|---|---|
| top 20% | **−15.71% [−27.76%, −3.29%]** |
| top 15% | **−22.93% [−38.12%, −7.73%]** |
| top 10% | −16.88% [−35.26%, +3.09%] |

**Where the model most disagrees with the market, the model is wrong** — CIs exclude zero
at the 15–20% cuts. This matches WP010 (deficit concentrates in high-line-movement
matches) and the edge-bucket table above (>40% claimed edge returns −14.47%).

**Fading it does not monetise that.** Pooled +3.02% [−9.05%, +15.82%], and positive in
only **4 of 10 seasons** — 2016–20 lost 24.9u, 2021–24 made 39.2u, 2025 lost again. A
random walk that happened to end up. The asymmetry is structural: backing loses ~16% but
fading recovers only ~3%, because the vig takes ~4.6% and with three outcomes "fade" is
spread across two alternatives rather than being a clean inverse.

## The decisive test: does the model add ANYTHING to the market?

Blend `P' = (1-w)*model + w*market` and find the best `w`. If the model held information
the market lacked, the optimum would sit below 1.

| subset | n | best w | gain vs market-only |
|---|---|---|---|
| **all matches** | 3,597 | **1.00** | +0.00000 |
| disagreement 25–75% | 1,798 | 0.95 | +0.00001 |
| disagreement top 25% | 899 | 1.00 | +0.00000 |
| market favourite weak | 1,187 | 1.00 | +0.00000 |
| market favourite strong | 1,187 | 0.95 | +0.00000 |
| *disagreement bottom 25%* | *900* | *0.00* | *+0.00077* |

**RPS falls monotonically as weight moves to the market**, and disagreement-conditional
shrinkage `w(d) = a + b·d` buys only **−0.00012** beyond a constant blend — from a
two-parameter grid search fitted in-sample.

The single exception is the bottom quartile of disagreement, where the model alone beats
the market (+0.00077). But that subset is *defined* as where the two already agree, so it
is where the choice matters least. The coherent reading: **the model is marginally better
than the market where they agree, and worse where they disagree.**

So the model is not "slightly behind" the market — on this evidence it is **dominated**:
its information appears to be a subset of the market's. That is a stronger statement than
the +0.0036 RPS gap alone conveys, and it is the real reason no betting strategy works.

## Reproducing

```bash
python work_products/wp024_market_backtest/backtest.py
```

Needs WP023's `checkpoints/` (champion, nutpie), WP021's `cv_shared_data_big.pkl`, WP008's `odds_raw.pkl`.
