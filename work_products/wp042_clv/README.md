# WP042 — Closing-line value: the high-power money test

**Verdict: the model has NEGATIVE edge. Its selections are measurably WORSE than
no-skill selection against the closing line, and the pattern is a textbook winner's
curse. This is the decisive money result and it is well-powered.**

## Why this test and not ROI

ROI grades a bet against the BINARY outcome, whose variance (~1.0 at even money, ~1.5-2.0
at 3-way odds) swamps any realistic edge — WP024's MDE was 6.66% on 1X2, WP041's 5.02% on
totals, both against professional edges of 1-3%. CLV grades the same bet against the
CLOSING PROBABILITY, a far lower-variance estimate of the truth:

    CLV = mean over taken bets of ( p_close_devig * odds_taken_at_open - 1 )

Measured power gain on the identical sample: **MDE ±0.30-0.49% vs ±5.19-5.81%, i.e.
11-17x more powerful.** This test can resolve 1-3% edges comfortably. The earlier ROI
nulls were never evidence of anything.

## The no-edge benchmark, without which the numbers are meaningless

CLV here uses the OFFERED odds, which carry the book's margin, so a bettor with ZERO skill
scores roughly MINUS THE OVERROUND, not zero. Reporting raw CLV against zero would have
presented the vig as negative skill. Benchmarks (price-weighted staking = no skill):

    Pinnacle open   -2.65% [-2.76%,-2.56%]
    bet365 open     -4.41% [-4.61%,-4.19%]

## Results — excess CLV over the no-edge benchmark

| arm | book | edge | bets | CLV | **excess** |
|---|---|---|---|---|---|
| champion | pin open | 0% | 4152 | −3.56% | **−0.91%** |
| champion | pin open | 2% | 3616 | −3.78% | **−1.13%** |
| champion | pin open | 5% | 2999 | −3.81% | **−1.16%** |
| champion | b365 open | 0% | 3613 | −5.23% | −0.82% |
| champion | b365 open | 5% | 2583 | −5.38% | −0.97% |
| + market channel | pin open | 0% | 3606 | −2.96% | −0.31% |
| + market channel | pin open | 5% | 2141 | −3.14% | −0.49% |
| + market channel | b365 open | 0% | 3032 | −4.47% | −0.07% |

**Every row is worse than no-skill selection**, and at MDE ±0.3-0.5% the champion's
−0.9% to −1.2% excess excludes zero.

## The mechanism: winner's curse, and the data shows it

The excess gets MORE negative as the edge filter TIGHTENS: −0.91% → −1.13% → −1.16% at
0% / 2% / 5% thresholds. That monotone pattern is the signature.

When a model has no real edge, filtering on "model EV > 0" does not select for skill — it
selects for **where the model's ERROR happens to be most positive**. Tighter filtering
means more adverse selection, so a bigger edge threshold makes things worse rather than
better. WP040 independently established the model adds nothing to the opening price, so
its deviations from that price ARE noise, and this is exactly what betting filtered noise
produces.

**This also reinterprets WP024, which reported that higher edge thresholds looked
slightly better.** That was noise in a test with 13x less power; the well-powered version
shows the opposite and gives a mechanism for it.

## The market channel reduces the damage, for the obvious reason

Its excess is −0.07% to −0.49% versus the champion's −0.82% to −1.16%: closer to
no-skill, i.e. less harmful. That is not an improvement in edge, it is the channel pulling
predictions toward market consensus, so there is less disagreement to be adversely
selected on. A model that exactly reproduced the price would have excess CLV of 0.00% and
zero edge. Consistent with WP040 and WP041.

## Line movement: no signal

    champion          d(close-open)/d(model-open) = +0.0080 [-0.0067,+0.0234]  corr +0.017
    + market channel                              = -0.0108 [-0.0323,+0.0102]  corr -0.019

The model's disagreements with the opening price do not predict which way the line moves.
So the closing-line-value strategy WP039 was conceptually aiming at does not exist here
either.

## What this settles

Betting this model on EPL 1X2 at opening prices would lose money **faster than betting at
random**, by about 1% of turnover. Not "no edge detected" — negative edge, measured with
a test that has the power to find a 1% effect.

Combined with WP040 (no information beyond the price) and WP041 (none on totals either),
the EPL betting direction is closed on measurement rather than on guesswork.

    python work_products/wp042_clv/clv_test.py

---

# CORRECTION (WP056 re-analysis): the −0.91% excess was a STAKING-UNIT artefact

The headline above — "every row is worse than no-skill selection", excess −0.91% to
−1.16%, read as a winner's-curse signature — does not survive. Two defects, both in the
BENCHMARK rather than in the model's CLV (that number, −3.56%, reproduces exactly).

## Defect 1: a per-bet quantity compared against a per-match benchmark

The model's CLV is a mean over BETS (4,152 bets from 3,597 matches; a match contributing
two selections counts twice). `bench_wt` is a mean over MATCHES, staking price-weighted
across all three outcomes. Those are different units. This file printed the per-bet
benchmark (`all-legs`, −3.35%) directly alongside and used the per-match one for `excess`.

Under Shin de-vigging the two benchmarks differ by 0.70pp, because equal-weighting the
legs over-weights longshots, where Shin assigns the margin most heavily. That 0.70pp IS
the finding. Re-expressed in a consistent per-bet unit:

| edge | model CLV/bet | vs all-legs (per bet) | vs side-matched |
|---|---|---|---|
| 0% | −3.56% | **−0.22% [−0.62%,+0.19%]** | −0.16% [−0.58%,+0.28%] |
| 2% | −3.78% | −0.44% [−0.89%,+0.02%] | −0.36% [−0.83%,+0.10%] |
| 5% | −3.81% | −0.47% [−0.96%,+0.02%] | −0.38% [−0.90%,+0.13%] |

Every CI now includes zero. `side-matched` is the tighter control: each bet is graded
against the mean CLV of flat-betting that same outcome across all matches, so it holds
the model's outcome MIX fixed as well as the match set.

## Defect 2: the conclusion is de-vigging-dependent, which was never tested

The closing price's de-vigged probability IS the measuring stick, so the de-vigger is not
a detail. Same model, same matches, same per-bet unit:

| grid | de-vig | excess vs side-matched @0% |
|---|---|---|
| TEST_LEN=3 | Shin | −0.16% [−0.58%,+0.28%] |
| TEST_LEN=3 | proportional | +0.37% [−0.02%,+0.76%] |
| horizon 1 | Shin | +0.05% [−0.30%,+0.39%] |
| horizon 1 | proportional | **+0.56% [+0.23%,+0.89%]** |

Only one of the four cells excludes zero, and it is the one using proportional de-vigging,
which is the WEAKER choice for 3-way markets with favourite–longshot bias. I nearly
reported that cell as "the model beats the no-edge benchmark at horizon 1". It is a
de-vigging artefact: under Shin the same data gives +0.05%.

## What actually holds

**CLV excess is NULL** — on both grids, under both de-viggers, against both benchmarks.
The model's selections neither beat nor lose to no-skill selection. The winner's-curse
mechanism and its "monotone in edge filter" signature are both withdrawn; the apparent
monotonicity was inside the noise band of a mis-united benchmark.

The PRACTICAL conclusion is unchanged and now better supported: absolute CLV is −2.1% to
−3.6%, i.e. you pay the margin and realise nothing back. This agrees with WP040's blend
test re-run at horizon 1 (optimal blend weight on the model w=0.045, and blending makes
out-of-sample log-loss WORSE by +0.00022) — the model carries no information the opening
Pinnacle price lacks. No edge, by two independent and now mutually consistent measures.

Lesson for future money tests: state the staking unit of the benchmark and of the
quantity, and make them the same; and report CLV under both de-viggers, because the
sign of a sub-1% excess depends on it.
