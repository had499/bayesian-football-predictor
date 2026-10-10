# WP041 — The model on other markets: totals and Asian handicap

**Verdict: NO EDGE on totals either, but a genuinely interesting finding on the way —
the model carries 4x more independent information about TOTAL GOALS than about match
OUTCOME. And the WP039 market channel actively DEGRADES totals, a cost invisible in 1X2
RPS.**

## Why look here

WP040 established a well-powered NO on 1X2 against the opening price. But every test in
this project had scored only the model's 1X2 COLLAPSE, discarding the total-goals
information in its scoreline grid — and the xG channel, the one established win, informs
goal RATES directly, which is exactly what a totals market prices.

**Data.** The same GitHub mirror WP008 used has over/under 2.5 and Asian handicap for
both opening and closing (football-data.co.uk itself is proxy-blocked; the mirror is not).
Totals exist only from season **2019**, so 2,478 matches over 7 seasons.

## The information finding

Blend weight `w` the model earns against the price (WP040's method, leave-one-season-out):

| market | champion `w` | blend gain vs price |
|---|---|---|
| 1X2, Pinnacle open | 0.046 | +0.00027 (worse) |
| 1X2, bet365 open | 0.098 | +0.00032 (worse) |
| **totals, Pinnacle open** | **0.203** | +0.00016 (worse) |
| **totals, Pinnacle close** | **0.244** | +0.00027 (worse) |
| **totals, market_avg** | **0.234** | **−0.00006** (marginal) |
| **totals, market_max** | **0.222** | **−0.00006** (marginal) |

**4-5x the 1X2 weight**, and the relative accuracy deficit is smaller too: on totals the
model is 0.44% behind the price in log-loss (0.67683 vs 0.67384) against 1.04% on 1X2
(0.95861 vs 0.94875). The scoreline grid really does hold information the 1X2 collapse
throws away.

**But it is not an edge.** The only negative blend gains are −0.00006 on `market_avg` and
`market_max` — negligible, and `market_max` is the series WP024 measured as 32%
arbitrage, i.e. not simultaneously available.

## ROI agrees, for what it is worth

| book | edge | bets | ROI | CI |
|---|---|---|---|---|
| pinnacle | 0% | 1763 | −0.41% | [−5.33%, +4.71%] |
| pinnacle | 5% | 853 | +0.43% | [−6.14%, +7.26%] |
| market_avg | 0% | 1472 | −0.87% | [−6.11%, +4.26%] |
| market_max | 2% | 1828 | +1.64% | [−2.85%, +6.02%] |

MDE is **5.02%**, better than 1X2's 6.66% — totals are near even-money so per-bet variance
is ~1.0 rather than ~1.5-2.0, and every match is bettable. Still cannot resolve 1-3%, so
this is corroboration rather than evidence. Disagreement quartiles show no pattern either
(Q2 +0.53%, Q3 −1.50%, Q4 +0.31%), so there is no concentrated edge where the model
differs most from the price — unlike the 1X2 pattern WP003/WP008 reported.

## The market channel damages totals

| | champion | + market channel (T100) |
|---|---|---|
| totals blend weight, Pinnacle | 0.203 | **0.087** |
| totals blend gain | +0.00016 | **+0.00060** |
| totals model-alone log-loss | 0.67683 | **0.67797** |

The channel is trained on rates recovered by inverting 1X2 prices under **Poisson
independence**, which compresses them to reproduce the market's draw price (`mkt_kappa` =
0.911, ~10% low). So the model's TOTAL-GOALS calibration gets pulled toward that bias
while its 1X2 accuracy improves. **This cost is invisible in 1X2 RPS and is a further
argument against adding the channel to the champion** — on top of P(improves) ≈ 90%,
4/10 seasons, and unfixed convergence degradation.

## Asian handicap: not testable here

Prices cover 93.6% of 2019+ matches, and `ah_line` / `ah_line_close` cover 100% (easily
missed — every BOOK-specific line column is 0% from 2019, so a first pass looked like the
line was unavailable). But only **24.6%** of lines are clean half-integers (654 matches);
whole lines push and quarter lines settle as split stakes, which does not fit a binary
blend test. 654 matches over 7 seasons is too thin to say anything, so this is left
untested rather than tested badly.

    python work_products/wp041_other_markets/fetch_odds.py
    python work_products/wp041_other_markets/totals_test.py
    python work_products/wp041_other_markets/totals_roi.py
