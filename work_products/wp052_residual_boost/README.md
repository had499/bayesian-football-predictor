# WP052 — Boosting the champion's residual

> ## ⚠ THE CONCLUSION BELOW IS WRONG — see premise3.py / WP052b
>
> The first pass used **16 features and ONE window**, and critically **no DEFENSIVE
> records at all**: it gave the learner the opponent's ATTACKING form (goals/xG scored)
> but never what anyone CONCEDES. To predict how many goals a team scores, the relevant
> opponent feature is its defensive record. That is a misspecification, not a thin
> feature set.
>
> Redone with 96 features — 8 quantities including conceded, windows of 3/10/25,
> volatility, short-minus-long trend, and home/away-split form:
>
>     permutation null (40 shuffles): mean -0.00197, 95% range [-0.00403,+0.00047]
>     OUR features (rich)   ridge     +0.00233  corr +0.0501   OUTSIDE null
>     OUR features (rich)   boosting  +0.00320  corr +0.0569   OUTSIDE null
>     + MARKET  [CEILING]   ridge     +0.00914  corr +0.1100   OUTSIDE null
>     p-value for OUR features: 0.000   (0 of 40 shuffles scored as well)
>
> **So the residual IS predictable from our own features**, at roughly a third of what
> the market achieves. The "information failure" verdict below was an artefact of MY
> feature set, and the composite-channel idea is back on.
>
> The first pass also reported POINT R^2 with ONE shuffle as its control and untuned
> boosting whose R^2 was negative on real and shuffled data alike — comparing two overfit
> models. The permutation null and regularisation sweep were added afterwards.


**Verdict: DEFINITIVE. No combination of the ~30 features this project has engineered
carries ANY information about what the champion is missing — they predict its residual
exactly as well as a SHUFFLED target. The market price predicts it at +0.01635 through
the identical pipeline. The missing information is external, not latent in our feature
space.**

## The question

~30 arms each tried to name one missing information source; almost all null. The market
channel then took 32% of the gap in one shot, because it imports an AGGREGATE rather than
guessing which piece is missing. So: were the nulls an AGGREGATION failure (each signal
too weak alone, but useful combined) or an INFORMATION failure (nothing there at all)?

If aggregation, then "include all thirty and ablate" would beat adding them one at a
time. This settles it without building anything.

## The method — boost the champion

Predict the champion's out-of-sample residual (`goals − lambda`) from our features,
leave-one-season-out, with both a linear (RidgeCV) and a non-linear learner
(HistGradientBoosting, depth 3). Our Bayesian model is the base learner; this asks
whether a second stage can find anything it left behind.

**Features restricted to what is knowable at the OPENING price**, or the answer would not
bear on the actual problem: prior-only rolling xG / goals / deep / shots for both the team
AND its opponent, stakes, rest, load, is_home, and the model's own lambda.
**Excluded:** `intact` (WP030 established it needs the match's own team sheet) and the
market price for THIS match, which is run separately as a CEILING rather than mixed in —
we already know the market beats us (WP040), so including it would guarantee success and
teach nothing.

## Result

    features                     learner       OOS R^2
    OUR features only            ridge        -0.00108
    OUR features only            boosting     -0.01288
    SHUFFLED target [control]    boosting     -0.01271      <- what ZERO looks like
    + MARKET price [CEILING]     ridge        +0.01635
    + MARKET price [CEILING]     boosting     +0.00154

n = 7,444 team-matches over 10 seasons, residual sd 1.170.

**Our features land exactly on the shuffled control.** Boosting −0.01288 vs −0.01271. A
learner with interactions and non-linearity finds in our thirty engineered signals
precisely what it finds in permuted noise.

**The market price sits an order of magnitude above the floor** through the same pipeline.

## What this closes

* **The composite-channel idea** — there is nothing to composite.
* **"Include all thirty and ablate"** — the saturated model would have nothing to ablate
  that a shuffled target does not also have. (It was also never cheaper: 31 sweeps vs 30,
  at the slowest possible configuration, and ablation is blind to collinear groups —
  WP021 measured shots and deep cancelling, so ablation would have hidden deep entirely.)
* **Any further arm built from our existing feature space**, one at a time or combined.
* And it retires the hypothesis that the ~30 nulls were an aggregation artefact. They
  were not. **WP007's L5 bundle losing to L0+xG was the early warning of the same thing.**

## What it supports

The market channel, by an independent method. WP039 measured it at −0.00083 RPS closing
32% of the gap; this shows it is the only thing in reach that predicts the residual at
all. Two different tests, same conclusion.

Ridge beats boosting on the market features (+0.0164 vs +0.0015), so the market signal
enters close to LINEARLY — a clean level shift rather than something with interactions,
which is also why boosting wastes capacity on it.

    python work_products/wp052_residual_boost/premise.py
