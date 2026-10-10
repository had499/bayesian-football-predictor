# WP054 — The GBM correction in-model: the Bayesian machinery declines it

**Verdict: NULL. +0.00002 [−0.00003, +0.00007]. `gbm_beta` fits at 5% of prior SD —
essentially where the five failed additive covariates died. WP053's post-hoc wrapper
gain exists ONLY in the two-stage form.**

## Why this was run

WP053 measured a GBM wrapper on the champion's lambdas at **−0.00062
[−0.00122, −0.00005]** (mature regime, config locked on disjoint windows, walk-forward
training). But a wrapper has no uncertainty: the Bayesian model says "lambda is
1.50 ± 0.14" and the GBM says "multiply by 1.03" as a bare point estimate, so the
corrected prediction is over-confident.

Entering the same signal as `theta += beta * z` (with `z = log(m)`, so `beta = 1`
reproduces the wrapper exactly — test-guarded) gives the correction a posterior AND lets
the model vote on it.

## Result

    champion  0.19692
    GBMCOV    0.19694    +0.00002 [-0.00003,+0.00007]   no clear difference
    mature windows (>=70): +0.00007 [-0.00001,+0.00014]
    LOSO: better in 1/10 folds

    gbm_beta  median +0.0268 = 5% of prior SD
              p5 -0.0981, p95 +0.1803, sign-consistent 79/115

For reference: `beta = 1` is the wrapper; congestion / absence / lineups / keeper /
intactness all fitted at ~2% of prior SD.

## The two architectures disagree, and that is the finding

| architecture | result |
|---|---|
| two-stage: Bayesian → GBM wrapper on its lambdas | **−0.00062** [−0.00122,−0.00005] |
| single-stage: the same signal as a fitted covariate | **+0.00002**, null |

Not a contradiction — different models. The wrapper fits the latents WITHOUT z and then
corrects the finished predictions. In-model, the latents are fitted WITH z available, so
they REALLOCATE: the same information that would have corrected the prediction instead
lets the latents fit the training rows better, and beta shrinks to nothing.

**This is the WP030 / WP031 / WP037 pattern for the fourth time** — the latents absorb a
signal during fitting rather than using it at prediction time. It is now the single most
reliable regularity in this project.

## What it implies

* WP053's gain is **real as measured** (out-of-sample, held-out windows, locked config)
  but exists only as a wrapper.
* The wrapper is therefore **overriding the posterior rather than informing it**. Whether
  that is acceptable depends on whether the goal is a scoring machine or a coherent
  probabilistic statement about football.
* The "no uncertainty" objection to the wrapper **stands unresolved** — the in-model route
  that would have fixed it does not deliver the gain.

## Predictions on record

I predicted beta well below 1 (0.2–0.4) and a smaller gain than the wrapper, reasoning
that z is built from the champion's own lambda plus features the latents already track.
Right in direction, wrong in magnitude: 0.027, not 0.2–0.4, and the gain is zero rather
than merely smaller.

    python work_products/wp054_gbm_covariate/prepare_data.py
    python work_products/wp054_gbm_covariate/run_cv.py --workers 6
