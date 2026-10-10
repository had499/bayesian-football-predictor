# WP039 — The betting market as a measurement channel

**Verdict: REAL but NOT ESTABLISHED, and blocked on convergence. −0.00079
[−0.00203, +0.00038] at weight 1.0, better in 10/10 leave-one-season-out folds, but
P(improves) ≈ 90% (below the ~96% bar that admitted `deep`) and `max rhat` reaches 1.412.
NOT added to the champion.**

| arm | weight | RPS | ΔRPS vs champion | LOSO | seasons | max rhat |
|---|---|---|---|---|---|---|
| champion | — | 0.19692 | — | — | — | ~1.10 |
| `M025` | 0.25 | 0.19647 | −0.00045 [−0.00141,+0.00051] | 9/10 | 4/10 | 1.201 |
| `M050` | 0.50 | 0.19618 | −0.00074 [−0.00183,+0.00034] | 10/10 | 4/10 | 1.279 |
| `M100` | 1.00 | 0.19613 | −0.00079 [−0.00203,+0.00038] | 10/10 | 4/10 | **1.412** |

115 windows, 3,780 matches, 345 fits, 0 divergences, 398 min.

## The idea

WP037 established that across ~25 arms **only LIKELIHOOD changes — new observations about
the same matches — have ever improved this model**, and WP038 closed the shot-level feed.
The one untried source of genuinely new information was the market itself.

Pinnacle's CLOSING 1X2 price for each match → Shin de-vig → invert a Poisson scoreline
model → an implied rate per side → observe it with the same Gamma machinery as xG.

**Why it is not leakage.** A measurement channel observes matches already played. The
channel is a `pm.Potential` on training rows, so predicting round R reads prices from
rounds < R only; verified that all 115 windows have `test_start == train_end + 1`, zero
overlap. `market_obs_w` is 0 on the 4.3% of matches with no price, so they contribute
exactly nothing rather than leaking a placeholder (test-guarded).

**Why it can beat the thing it learns from.** WP010 measured closing lines to be sharper
than opening ones. This distils closing-quality information about the PAST in order to
price future OPENING lines — that asymmetry is the exploitable part.

## Premise and control — both strong

    OOS R^2 predicting next-match goals, leave-one-season-out:
      xG only (incumbent)            +0.07136
      xG + goals        [CONTROL]    +0.07268   +0.00132
      market CLOSING only            +0.08496   +0.01360   <- beats xG ALONE
      xG + market closing            +0.08721   +0.01586

+0.0159 is **3.3x the +0.00484 at which WP019's raw shot count failed as a channel**
(WP038's calibration anchor) and 12x the known-real control.

**Negative control (window 70).** Shuffling the market rates across training matches
within season destroys the entire gain, and `mkt_shape` collapses from 74 to 4.7 — the
model detects on its own that shuffled rates are uninformative. Opening lines score
slightly worse than closing (−0.00959 vs −0.01016), the ordering WP010 predicts.

    real closing     -0.01016   mkt_shape  74.16
    SHUFFLED         +0.00169   mkt_shape   4.72
    opening line     -0.00959   mkt_shape  97.76

So the effect is match-specific information, not a structural artefact.

## I overstated it twelvefold

The window-70 smoke read **−0.010**; the 115-window truth is **−0.00079**. I flagged
before launching that single-window reads are unreliable below ~13x MDE (WP035's
calibration) and that −0.010 — 3x the entire remaining gap — was implausibly large.
It was. Window 70 is simply a favourable window. **The smoke/control framework is
sound for establishing MECHANISM and worthless for MAGNITUDE.**

## The blocker: convergence

`max rhat` rises 1.201 → 1.279 → 1.412 with the weight, against the champion's ~1.10.
An arm whose posterior is not reliably sampled should not ship regardless of its RPS.

The likely cause is diagnosable rather than fundamental: **`mkt_shape` fits at 58–120
against a `Gamma(2, 0.5)` prior whose mean is 4.** A shape of 120 means the channel
believes the market's implied rate to within ±9%, which makes the likelihood extremely
stiff — bad geometry, and almost certainly over-trusting the market on training rows.
`mkt_kappa` = 0.911 is behaving exactly as intended, absorbing the ~10% level offset
that Poisson-independence inversion creates.

### WP039b: tightening the shape prior did NOT fix it

Tried `Gamma(8, 2)` (same mean 4, far less tail) as arm `T100`, at weight 1.0:

    arm                      median      p90      max   n>1.1   n>1.2   med secs
    champion (W075)           1.032    1.067    1.123       3       0        151
    market M100               1.072    1.152    1.412      32       7        502
    market T100 (tight)       1.073    1.183    1.309      38       9        525

Only the single worst window improved (1.412 -> 1.309). The MEDIAN is unchanged and
windows above 1.1 went UP, 32 -> 38. RPS is the same within noise: T100 **-0.00085
[-0.00207,+0.00031]**, 10/10 LOSO, 4/10 seasons. The shape still lands at 104 (vs 120),
so the prior barely moved the fit.

**A METHOD ERROR WORTH RECORDING.** I screened the shape settings on ONE window and
argued that was legitimate because "rhat is a property of the posterior GEOMETRY, not of
the 74-match test set, so a single window reads it reliably". **That reasoning is wrong.**
rhat is itself estimated from finite chains, so it carries sampling noise like any other
statistic. On window 45 the tight prior read 1.412 -> 1.053, which looked like a clean
fix and was mostly noise. One window is unreliable for rhat for the same reason it is
unreliable for RPS. (It took screening the WORST window to get even that far -- the first
screen used window 70, where every setting converges fine, and was uninformative.)

### Where this leaves the channel: real, promising, NOT shippable

* −0.00085, better in **10/10** leave-one-season-out folds, ~27% of the remaining gap
* P(improves) ≈ 90%, **below the ~96% bar** that admitted `deep`
* better in only **4/10** individual seasons
* **convergence degraded and unfixed**: 38/115 windows above rhat 1.1 vs the champion's
  3/115
* **3.5x slower** per fit (525s vs 151s)

**Untried remedies, in order of expected value:**
1. **More tune steps.** The standard answer to a stiff posterior, and the project has
   never moved off 600/600. ~7h for a 115-window sweep at the observed per-fit cost, so
   worth a decision rather than an assumption.
2. **Lower the weight.** `M025` already converges far better (median 1.057, 14/115 above
   1.1) for about half the gain (−0.00045). There is a real trade-off curve here between
   convergence and effect size, and 0.25-0.5 may be the honest operating point.
3. **Reparameterise the channel.** The stiffness is intrinsic: the market's implied rate
   tracks the model's lambda to ~±10%, so the Gamma likelihood is sharp almost by
   construction. Observing a TRANSFORM (e.g. the log-rate difference between the sides,
   which is what the market actually prices) might be better conditioned than observing
   each side's rate separately.

## Prediction accuracy is not betting edge

A caveat that belongs with the result, not buried: this channel pushes our predictions
TOWARD market consensus. That raises RPS and simultaneously shrinks the disagreements
that betting profit depends on. A model reproducing the closing line exactly would have
superb RPS and ZERO edge.

What survives is specifically the open-to-close gap: where we disagree with today's
opening price we are largely predicting that the LINE WILL MOVE — a closing-line-value
strategy, which is real but a different thing from "the market is wrong about this team".
Two consequences: we would inherit the market's own biases (exactly the mispricings an
independent model could exploit), and the edge requires actually betting at opening
prices. **Testable** with WP024's backtest machinery — though that test's MDE was 6.66%
against professional edges of 1–3%, so it needs its power fixed first.

    python work_products/wp039_market_channel/premise.py
    python work_products/wp039_market_channel/prepare_data.py
    python work_products/wp039_market_channel/control.py     # the shuffle control
    python work_products/wp039_market_channel/run_cv.py --workers 6
    python work_products/wp039_market_channel/analyse.py
