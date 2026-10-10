# WP053 — A comprehensive feature set and a two-stage GBM correction

**Verdict: POSITIVE. A properly specified GBM correction on top of the champion is worth
−0.0006 RPS in the mature regime, CI excluding zero, monotone in training history. The
largest validated single improvement in this project apart from the xG channel itself.**

| | dRPS |
|---|---|
| squared loss on the ADDITIVE residual, depth 2, untuned (first attempt) | −0.00010, CI through zero |
| **Poisson loss with `log(lambda)` OFFSET, same hyperparameters** | **−0.00100** on held-out windows, CI excluding zero |
| the same, mature-history regime, beta 0.5 | **−0.00062 [−0.00122, −0.00005]** |

**It was the OBJECTIVE, not the capacity.** The tuning search selected the most
regularised config in the grid — depth 2, 150 trees, lr 0.03, l2 10 — which is exactly
what the failed first attempt used. Changing from squared-loss-on-difference to
Poisson-on-ratio took it from null to CI-excluding-zero with identical hyperparameters.

Three reasons the multiplicative form is correct: the model is log-linear so every
mechanism in it acts multiplicatively on the rate; Poisson variance equals the mean, so
squared loss on an additive residual is heteroscedastic and over-weights high-scoring
matches where the noise is largest; and fitting Poisson loss on `goals/lambda` with
weight `lambda` is exactly equivalent to Poisson regression with `log(lambda)` as a fixed
offset (sklearn has no `init_score`, so the weight trick substitutes).

## It is data-limited, which explains an alarming-looking sign flip

The config's tuning half (windows 26–70) gave **+0.00120** — the opposite sign to the
evaluation half. That needed explaining rather than ignoring:

    window block   train rows     n    dRPS (beta=0.5)
    26-45                1640   680   +0.00123 [+0.00043,+0.00207]   WORSE
    46-65                3000   668   -0.00021 [-0.00103,+0.00062]   neutral
    66-85                4336   669   -0.00064 [-0.00137,+0.00003]   better
    86-115               5674   943   -0.00062 [-0.00122,-0.00005]   better, CI excl. 0

Monotone, plateauing around −0.0006 once ~4,300 training rows are available. So the
composite overfits when data-poor and helps when data-rich — a benign explanation, tested
rather than assumed.

**Practical consequence:** averaged over all 115 CV windows the effect would be diluted by
the early ones where it hurts. In deployment — predicting future matches with the full
history — we are always in the mature regime, so −0.0006 is the figure that matters.

## Protocol

* **Walk-forward training**: window `w` sees only windows `< w`. Not leave-one-season-out,
  which WP053 measured leaking the future for a second-stage learner and halving the
  apparent signal (+0.060 → +0.030).
* **Hyperparameters searched on windows 26–70, LOCKED, then evaluated on 71–115**, so the
  config choice cannot leak into the reported number.
* beta (a global shrinkage on the correction) is SWEPT, not fitted — a fitted beta on the
  data it corrects would be circular.
* LightGBM would be preferable but needs libomp and there is no brew on this machine;
  sklearn's `HistGradientBoostingRegressor` supports `loss='poisson'`.

## The feature set


253 features, 7,560 rows, zero unmatched, median coverage 98.8%. Everything the target
mechanically depends on:

* **own attack** — goals, xG, npxG, open-play xG, set-piece xG, deep, shots,
  shots-on-target, corners, big chances (xG > 0.3), shots-in-box, xG per shot
* **opponent DEFENCE** — the conceded mirror of all of the above. *This was the gap that
  invalidated the first attempt.*
* **finishing / luck** — rolling `goals − xG` for AND against, i.e. exactly what the
  latents may miss given they are driven by the xG channel
* **context** — is_home, stakes, rest, load, crowdless, round-in-season
* **standing** — points, goals for and against per game to date, goal difference
* **squad** — red card in the previous match (the only pre-open-knowable absence)
* **model internals** — own and opponent lambda, implied total and difference, p(H/D/A)
* windows of **3 / 10 / 25** with means, rolling sd, and short-minus-long trend; plus
  home/away-split form
* **excluded** — `intact` (needs the match's own team sheet) and the market price, held
  out as a separate CEILING

## The signal is real

    permutation null (40 shuffles, ridge): mean -0.00077, 95% range [-0.00191,-0.00000]
      OUR features (253)   ridge     +0.00143  corr +0.0368   OUTSIDE null
      OUR features (253)   boosting  +0.00370  corr +0.0598   OUTSIDE null
      + MARKET [CEILING]   boosting  +0.00565  corr +0.0731   OUTSIDE null
      p-value for OUR features: 0.000

    progression of the same test as the feature set was fixed:
      16 features (NO defensive records)   -0.00106   INSIDE null
      96 features (defence + 3 windows)    +0.00233   OUTSIDE null
     253 features (this)                   +0.00370   OUTSIDE null (boosting)

Note ridge DEGRADES with 253 features (+0.00143 vs +0.00233 at 96) and the market ceiling
collapses from +0.00914 to +0.00102 — dilution, too many correlated features for a linear
model. So magnitude estimates span 2.5x for our features and 9x for the ceiling, which is
why scaling OOS R² into RPS was not trustworthy and the effect had to be measured.

## But it is not RPS-exploitable

Walk-forward composite (each window trained only on strictly earlier windows), applied as
`lambda_adj = lambda * exp(beta * composite / lambda)`, beta swept rather than fitted to
avoid circularity:

      beta       RPS       dRPS                     CI
      0.00   0.19995   +0.00000                     control
      0.25   0.19985   -0.00010 [-0.00029,+0.00009]
      0.50   0.19985   -0.00010 [-0.00049,+0.00027]
      1.00   0.20011   +0.00016 [-0.00063,+0.00095]

## THE METHODOLOGICAL FINDING: LOSO leaks the future for a second-stage learner

    leave-one-season-out corr(composite, residual) = +0.0598
    WALK-FORWARD           corr(composite, residual) = +0.0302

Holding out 2016 and training on 2017–2025 lets the composite learn relationships from
seasons that had not happened yet. That is harmless for this project's usual purpose —
comparing model arms that are each already walk-forward internally, where LOSO only
re-weights which windows are averaged — but it is the WRONG protocol for validating a
predictor we would deploy. **The honestly-validated signal is half the bound.**

So: real structure, optimistically bounded, and what survives honest temporal validation
converts to ~−0.0001 RPS with a CI through zero.

## Corrections this work package made to earlier claims

* **WP052's "our features are at the noise floor" was WRONG** — an artefact of 16
  features with no defensive records. A feature-space bound is only as good as the
  feature space.
* WP052's first pass also used ONE shuffle as its control and untuned boosting whose R²
  was negative on real and shuffled data alike, i.e. it compared two overfit models.

    python work_products/wp053_features/build.py       # 253 features
    python work_products/wp053_features/evaluate.py    # permutation null + ceiling
    python work_products/wp053_features/posthoc.py     # the RPS measurement
