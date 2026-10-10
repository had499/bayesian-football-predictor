# WP040 — Does the model have a betting edge? A test with actual power

**Verdict: NO. The model adds NOTHING to the opening price out-of-sample, and this is a
well-powered negative rather than an absent result. The WP039 market channel improves
standalone accuracy and does NOT create an edge.**

## Why not just fix WP024's ROI test

WP024's MDE was 6.66% against professional edges of 1–3%, so it could not have detected a
real edge either way. **That is not fixable by a better estimator.** ROI is
`sum(pnl)/sum(stake)` over ~1,000 flat-stake bets at mean odds ~3, and the variance is
dominated by the binary win/lose outcome, which carries no information about whether an
edge exists. Resolving 1–3% needs ~10–50k bets — a decade of multi-league data. The
sample is the binding constraint, not the method.

## The powerful alternative

"Is there an edge" is the same question as "do the model's DISAGREEMENTS with the price
carry information". That is testable on EVERY match instead of a filtered subset of bets,
and never discretises into win/lose:

    p_blend = normalise( p_market^(1-w) * p_model^w )

`w` is fitted by log-loss, **leave-one-season-out**. `w = 0` means the model adds nothing
the price lacks.

## Results — all at OPENING prices, what a bettor can act on

| | blend weight `w` | OOS logloss vs market alone |
|---|---|---|
| champion, Pinnacle open | 0.046 | **+0.00027 (worse)** |
| + market channel, Pinnacle open | 0.056 | **+0.00023 (worse)** |
| champion, bet365 open | 0.098 | +0.00032 (worse) |
| + market channel, bet365 open | 0.087 | +0.00019 (worse) |
| champion, market_avg | 0.020 | +0.00016 (worse) |
| + market channel, market_avg | 0.020 | +0.00014 (worse) |
| **CLOSING price, Pinnacle open** | **1.000** | **−0.00313 (BETTER)** |

Blending the model into the price makes out-of-sample log-loss **worse in all six cases**.
The fitted weights of 0.02–0.10 do not survive the fold they were not fitted on.

## The positive control is what makes this interpretable

A null is only meaningful if the test can detect a real improvement. Given one it must
find — Pinnacle's CLOSING price blended into its own OPENING price, which WP010 measured
as strictly sharper — the test returns **w = 1.000 in all 10 folds** and a log-loss gain
of **−0.00313**. It has power, and plenty of it.

*(A second control, blending a synthetically sharpened market `p^k` into the opener, was
FLAWED and proves nothing: sharpening an already-calibrated forecast degrades it, so all
four rows were worse in isolation and "NOT detected" is the correct answer rather than a
power failure. Annotated in the script and kept for the record.)*

## What this settles

**1. There is no edge against the opening line.** Not "we cannot tell" — WP024's
underpowered conclusion — but a measured no, from a test that demonstrably detects a real
edge when one exists.

**2. The market channel improves PREDICTION, not EDGE.** Its standalone log-loss improves
0.95861 → 0.95538 (and RPS by −0.00085), so WP039's gain is real. But its blend weight
barely moves and blending still loses. **The information it adds is information the market
already had** — exactly the failure mode predicted when WP039 was launched: a channel that
pulls predictions toward consensus raises accuracy while shrinking the disagreements
profit depends on.

**3. It reframes the whole project's remaining gap.** WP009 concluded the residual is
missing information; ~27 arms confirmed it from the other direction; this now confirms it
from the market's side. The closing price contains something real (w = 1.0) that neither
our latents nor any channel tried so far can reconstruct.

## What is NOT excluded

* A **concentrated** edge in some subset (high-disagreement matches, specific price
  ranges). Blending is a global average test. WP024's edge-threshold ROI already probed
  this and found nothing, but with low power.
* Markets other than 1X2 — Asian handicap, totals, correct score — where the model's
  scoreline distribution might be better exploited than its outcome probabilities.
* Softer books than the three tested, or in-play.

    python work_products/wp040_edge_power/blend_test.py
