# WP064 — can `tune` be halved? NO: 25% less time, 25% less ESS, zero efficiency gain

Motivated by WP063's incidental finding that NUTS tree depth is EXACTLY 6 in 2,400/2,400
draws with `maxdepth_reached` at 0%, suggesting 600 tuning draws were overkill. Tuning is
half the gradient evaluations (600 tune + 600 draws), so halving it looked like ~25% off
every sweep. `SAMPLER_FAST` (300/300) had been rejected, but that cut draws AND tune
together, so tune alone was genuinely untested.

Three windows, both settings, run SERIALLY on an idle machine for clean timings.

| window | t=600 | t=300 | time saved | depth | **min ESS** | rhat |
|---|---|---|---|---|---|---|
| 100 | 45s | 33s | 26.8% | 6.00/6.00 | 159/158 | 1.019/1.023 |
| 200 | 69s | 55s | 20.4% | **199/109** | 6.00/6.00 | 1.019/1.037 |
| 300 | 114s | 82s | 28.1% | **100/81** | 6.00/6.00 | 1.021/1.045 |
| total | 229s | 171s | **25.5%** | | | |

## Verdict: REJECT

**Effective samples per second: 2.00 at tune=600 vs 2.04 at tune=300 — identical.** The time
saving is exactly offset by ESS loss. You get 25% more fits each carrying 25% less
information. It is not a speedup, it is a different point on the same efficiency curve. And
rhat degrades consistently (1.019→1.037, 1.021→1.045), so it is a quality regression on top
of a non-gain. `tune` stays at 600.

## The reasoning error worth keeping

I argued the experiment was **self-diagnosing**: if adaptation failed, trees would get deeper
and the failure would show up in the depth statistic rather than hiding in the results. That
was wrong, and instructively so.

Depth stayed at exactly 6 with `maxdepth_reached` at 0% — **step-size adaptation did
succeed**. But ESS also depends on the **MASS MATRIX**, which for a 29,000-dimensional model
genuinely needs the draws to estimate. A worse metric produces more correlated steps *at the
same tree depth*. So depth is sensitive to step-size failure and **completely blind to metric
degradation**.

**Lesson: tree depth is a necessary but not sufficient adaptation diagnostic. Judge sampler
budget changes on ESS PER SECOND, never on wall time alone, and never on depth alone.**

This also retrospectively explains WP023's result: `SAMPLER_FAST` (300/300) cost |ΔRPS| up to
0.00042 not only because of fewer draws but because the metric was under-adapted too.
