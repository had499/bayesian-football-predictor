# WP043 — The model on the Russian Premier League

> # ⚠ THE RESULT BELOW IS VOID — IT WAS A LEAK. Rerunning.
>
> **`prepare_model_data` splits train/test on `round`, not `cum_round`.** The EPL frame
> happens to carry a CUMULATIVE counter in `round` (1–416, identical to `cum_round`).
> My RFPL frame put the WITHIN-SEASON round (1–31) in `round` and the cumulative counter
> in `cum_round`, so every match passed `round <= train_end` (which reaches 349) in every
> window. **The model trained on its own test rows.**
>
> The signature was there to be seen: `n_train` was a constant **2,880** (the whole
> dataset) in all 98 windows, where the EPL grows 780 → 4,548. I never checked it.
>
> Everything that looked like robustness was the leak: beating a calibrated closing line,
> profitable in 10/10 seasons, an edge in EVERY odds band, surviving the removal of
> longshots. A real edge does not look that good, and I said so repeatedly without
> drawing the obvious conclusion.
>
> Fixed: `round` is now the cumulative counter, with an assertion that the training
> filter actually excludes a growing share of rows (window 1 keeps 960 of 5,760; window
> 98 keeps 5,710). Window 50 now trains on 1,695 rows instead of 2,880 and scores
> 0.20170 instead of the leaked figure. The sweep is rerunning.
>
> **Keep in mind reading on: the numbers in the ORIGINAL sections below are
> contaminated. The clean rerun is immediately after this block.**

---

# CLEAN RESULT (after fixing the leak)

**Verdict: NO EDGE. The model LOSES to the Russian closing line by +0.00336
[+0.00116, +0.00552], and blending it into that price still makes log-loss slightly
worse. The xG channel does transfer. The "weaker venue" thesis is partially supported
but nowhere near enough to profit.**

    xG channel    L0 0.20223 -> XG75 0.20047   -0.00177 [-0.00298,-0.00052]  CI excl 0
                  [EPL: -0.00258 [-0.00414,-0.00095]]

    vs RFPL CLOSE model 0.20118  market 0.19783   +0.00336 [+0.00116,+0.00552]
    blend         w = 0.145 [0.092,0.210]   logloss +0.00011 (WORSE)

## What IS supported

| | deficit vs the price |
|---|---|
| model vs EPL **closing** line | +0.0139 |
| model vs EPL **opening** line | +0.00315 |
| **model vs RFPL closing line** | **+0.00336** |

The model sits about as far behind the Russian CLOSE as behind the English OPEN -- a
**4x smaller deficit against a closing line**. Its blend weight is **3x the EPL's**
(0.145 vs 0.046), so it genuinely carries more independent information relative to this
market. A thinner market does narrow the gap substantially; it does not close it.

The model is also absolutely worse here (0.20118 vs 0.19692): 2,880 matches rather than
4,560, 98 windows rather than 115, and no `deep` or `stakes_effort` channels.

## What would be needed

The gap to beat is now **+0.00336**, not +0.0139. Either a market weaker still, or a
model better by that much in a league where we have 37% less data. Neither is obviously
out of reach, and the +0.00336 figure is the one to aim at.

---




## Why RFPL

WP040/WP041/WP042 closed EPL betting on measurement: no information beyond the opening
price, none on totals, and NEGATIVE edge via winner's curse. The model itself is decent
— the problem is the VENUE. RFPL is the one league where Understat supplies the xG data
our only working channel needs, while market attention is far lower.

## Setup

* Understat RFPL 2014–2025, 2,880 matches, 32 teams. 98 windows, STEP = TEST_LEN = 3,
  first test season 2016 (matching the EPL grid). 2,400 test matches.
* Model: **L0 + xG channel at weight 0.75** (WP032's tuned value), not the full EPL
  champion — `deep` and `stakes_effort` went in on expected value and would each need
  new RFPL data.
* Odds: football-data.co.uk's extra-league file (**it was recorded as proxy-blocked;
  it responded fine on 2026-10-05**). **CLOSING odds only** — so no CLV, and every bet
  is struck at the sharpest price of the market's life.

## Results

**1. The xG channel transfers almost exactly.**

    L0     RPS 0.19309
    XG75   RPS 0.19075      -0.00234 [-0.00349,-0.00124]   CI excludes zero
    [EPL:                   -0.00258 [-0.00414,-0.00095]]

**2. The model beats the RFPL closing line.**

    model  RPS 0.19137
    market RPS 0.19783  (Pinnacle CLOSING)
    gap    -0.00646 [-0.00847,-0.00448]   MODEL AHEAD
    [EPL: +0.00315 behind the OPEN, +0.0139 behind the CLOSE]

**3. The blend test gives the market zero weight.** `w = 1.000` in all 10 seasons,
log-loss 0.98330 → 0.96295 (−0.02035). On the EPL `w` was 0.046 and the blend was worse.

## Why this is coherent rather than a bug — the control that matters

Skill over each league's OWN base rates:

| | skill |
|---|---|
| EPL market (Pinnacle close) | **+0.04077** |
| our model, EPL | +0.03576 |
| **our model, RFPL** | **+0.03650** |
| RFPL market (Pinnacle close) | **+0.02954** |

**Our model's skill is constant across leagues (+0.0358 vs +0.0365).** What changes is the
opposition: the EPL market beats us, the RFPL market (28% less skilful) does not. A join
bug or a leak would not produce a model whose skill reproduces its EPL value to three
decimals. This is the "venue, not model" thesis measured.

**And the odds are genuine.** RFPL closing is calibrated to ±0.008–0.019 by market-
probability bin, essentially identical to the EPL's ±0.001–0.017. The market is sharp,
just less informed.

## The ROI, and why I do NOT believe it yet

    pinnacle CLOSE, flat stakes, edge>0:  +16.85% [+11.32%,+22.76%]   n=2619 bets
    model's OWN expected ROI +17.83%  vs realised +16.85%  (they agree)
    realised win rate 0.358 vs model-predicted 0.346 (slightly UNDER-confident)
    staking every leg: -4.56% vs the -3.33% de-vigged margin (accounting is sound)

The P&L arithmetic checks out and the realised return matches the model's own
expectation. **It is still implausible.** Professional edges are 1–3%; +17% at CLOSING
prices would be extraordinary. Two specific reasons for doubt:

* **A +0.0065 RPS edge should not yield +17% ROI.** The leverage comes from longshots
  (mean odds taken 4.411), where RPS is insensitive and ROI is not — so it is
  arithmetically possible, but it means the entire result rests on a thin slice of
  low-probability bets.
* **The model is poorly calibrated at the extremes.** Binned by its own prediction it
  says 0.178 where reality is 0.108, and 0.644 where reality is 0.701 — too flat at both
  ends. A too-flat model betting longshots should LOSE, so the mechanism of the gain is
  not understood.

## What must be checked before this is believed, let alone acted on

1. **Per-season ROI stability** — is it one freak season? (the script errored on this;
   it is the first thing to fix)
2. **Why the model is uncalibrated yet profitable** — resolve the contradiction above.
3. **Stake limits.** RFPL at Pinnacle closing may accept very little; a 17% edge on £20
   a match is not a business.
4. **A second league as replication** (Ligue 1, Serie A) — if the effect is real it
   should appear wherever market skill is low, and NOT appear in the EPL.
5. The `market_max` series shows a 0.02% overround, i.e. pure arbitrage, so those rows
   are untradeable (the WP024 finding again).

    python work_products/wp043_rfpl/fetch.py
    python work_products/wp043_rfpl/prepare_data.py
    python work_products/wp043_rfpl/join_odds.py
    python work_products/wp043_rfpl/run_cv.py --workers 6
    python work_products/wp043_rfpl/analyse.py
    python work_products/wp043_rfpl/roi.py
