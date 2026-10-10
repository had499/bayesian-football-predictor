# WP061 — availability IN-MODEL as a fixed-coefficient Poisson offset: WORKS, but EQUIVALENT to post-hoc

686/686 fits, 0 failed, 154 min. Baseline wp055 `H1_CHAMP`, whose exact reproduction on this
data file was verified before launch (lambda identical to 6 d.p.).

## Why this run existed

WP059 showed published injury data works POST HOC. I argued post-hoc was the right home,
citing four in-model failures. The user pushed back, correctly: that evidence is about
FITTED coefficients being crushed by ~29,000 latents, which is not an argument against being
in-model. An offset with a FIXED k has no parameter to crush, and it buys a SECOND mechanism
post-hoc cannot touch — it keeps an injury-driven dip out of the latent at fitting time,
rather than letting the AR(1) walk propagate it forward as a permanent quality change.

    lambda = exp(theta + log(1 - k*out_own) + log(1 + k*out_opp))

## Result

| arm | k | RPS | dRPS vs OUTCOME | dRPS vs CLOSE |
|---|---|---|---|---|
| champion | — | 0.19657 | — | — |
| AVL020 | 0.20 | 0.19641 | −0.00016 [−0.00038,+0.00006] | **−0.000094 [−0.000125,−0.000064]** |
| AVL035 | 0.35 | 0.19637 | −0.00019 [−0.00058,+0.00018] | **−0.000069 [−0.000123,−0.000016]** |

On windows >200, the range directly comparable to WP059's held-out post-hoc figure:

| applied | outcome | close |
|---|---|---|
| IN-MODEL offset (k=0.20) | −0.00021 [−0.00056,+0.00013] | −0.000128 [−0.000176,−0.000079] |
| POST-HOC multiplier (k=0.20) | −0.00024 [−0.00063,+0.00015] | −0.000113 [−0.000169,−0.000059] |

**Identical within noise.** Sampler health is fine — median rhat 1.037 against the champion's
1.038, 9/343 windows above 1.10 (champion: 14/343), zero divergences. So the offset does NOT
degrade conditioning, unlike the market channel's 1.374. LOSO: better in 7/10 seasons.

## TWO OF MY PREDICTIONS FAILED

1. **In-model does NOT beat post-hoc.** The latent-hygiene mechanism I argued for contributes
   nothing measurable. The entire benefit is the prediction-time adjustment.
2. **In-model does NOT want a larger k.** I predicted it would, since cleaning the latents is
   an extra channel of benefit. k=0.35 is no better than 0.20 (and worse on the full grid).

## What that means

The AR(1) walk evidently does NOT meaningfully mis-propagate injury dips — consistent with
`feedback-dont-clean-persistent-nuisance`, which I had argued pointed the other way here.
Cleaning the latents of a nuisance bought nothing for the fifth time.

**So POST-HOC is the right home after all, but for a different reason than I gave.** My
original argument (latents crush covariates) was the wrong argument — the user was right to
reject it. The correct argument is COST: identical gain, zero MCMC, no sampler risk, and it
composes with the GBM wrapper (WP060). The in-model version is defensible but buys nothing
for ~5 hours of compute per configuration.

## Engineering note

The offset lives in ONE shared function, `components.avail_offset_terms(out_h, out_a, k, log)`,
called by build_model with `pm.math.log` and by predict.py with `np.log`. theta was previously
duplicated by hand across those two files, which has caused two real bugs. Three earlier
attempts to TEST the equivalence instead were all bad tests: `model.initial_point()` is RANDOM
here (two builds of identical data gave lambdas differing by 0.72), and compiling a
deterministic over `model.value_vars` with `on_unused_input="ignore"` silently yields a graph
that does not depend on its inputs — it "passes" while measuring nothing. Sharing the
implementation removes the failure mode instead of watching for it. 7 tests, full suite 236.
