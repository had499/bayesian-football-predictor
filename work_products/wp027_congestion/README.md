# WP027 — Fixture congestion, measured properly

**Status: complete. Verdict: NULL, and definitively so — the tightest interval in the project.** Rest days across all competitions: **−0.00000 RPS [−0.00004, +0.00003]**. 14-day fixture load: **−0.00002 [−0.00006, +0.00001]**. Both coefficients fit at 2% of one prior SD.

**The champion is unchanged.**

## Why this was worth doing

WP009 tested rest days and found nothing, but using a **league-only** measure. A team playing Thursday in the Europa League then Sunday in the league looked identically rested to one with a clear week. The BACKLOG recorded that null precisely: *"league-only rest-day differential doesn't help, **not** congestion doesn't matter"*, and flagged cup/European fixture dates as the proper fix.

It is also the one open idea whose data requirements the accessible sources actually meet. Rest days need only fixture **dates** — no xG, no results, no opponent latent — so openfootball's lack of xG doesn't matter. And cup/European fixtures are absent from Understat entirely, so this is information the latent walk could never have absorbed from past league results.

## The measurement was genuinely broken

Rebuilding rest from the complete calendar (league + FA Cup + EFL Cup + CL/EL/UECL):

| | |
|---|---|
| matches where rest differs from league-only | **27.9%** |
| home sides on ≤3 days' rest: league-only → all-comps | 981 → **1,444** (+47%) |

So WP009's measure was wrong for a quarter of matches and missed nearly half the short-rest cases. The BACKLOG's caveat was correct.

## Results

115 windows. Headline is the **complete-calendar** seasons: openfootball has domestic cups only from 2018-19 and the Europa League only from 2020-21, so for Understat seasons 2016/2017/2025 — **1,140 of 4,560 matches, 25% of the grid** — the covariate reverts to roughly WP009's broken measure. Pooling a well-measured covariate with a badly-measured one attenuates the coefficient and would misattribute the null.

| | RPS (2018–2024) | Δ vs champion |
|---|---|---|
| champion | 0.19862 | — |
| `G_rest` | 0.19862 | **−0.00000** [−0.00004, +0.00003] |
| `H_load` | 0.19860 | **−0.00002** [−0.00006, +0.00001] |

Full grid agrees (−0.00000 / −0.00001), so the mismeasurement wasn't hiding anything.

Tested as **separate arms** because `corr(rest, load) = −0.689` and WP011's crux test looked significant purely because two features at r=0.76 split into cancelling coefficients (BACKLOG caution 1).

## Why the intervals are 20x tighter than usual

±0.00004 against the project's usual ±0.00095. The coefficients explain it: **+0.0025 (rest) and +0.0015 (load), both 2% of one prior SD.** Handed the covariate, the model declines it — the same "declined to use it" signature as the original rolling-mean xG covariate (`beta_xG ~ HalfNormal(0.1)`, fitted at ~zero because the latents already explained it). For contrast, the WP025 Elo anchor fitted at **+0.169 = 169% of its prior SD, positive in 115/115 windows** — what a covariate the model actually uses looks like.

A tight interval around zero and an unused coefficient are the same fact seen twice. And the tightness is the real result: congestion is worth **less than 0.00006 RPS**, i.e. under 2% of the gap to the opening line. WP009's null was ambiguous; this one is not.

## Why it was not added despite positive expected value

`H_load` has E[gain] = −0.00002 and P(improves) ≈ 87%, so by the expected-value standard used to include the deep channel it technically qualifies. It was still rejected:

* The **Elo anchor was declined at −0.00021** for depending on an external source. Load is a tenth that effect with a worse dependency: openfootball scraping, three parser bugs, a hand-maintained 35-club alias map, and 25% incomplete coverage.
* **P ≈ 87% sits inside the noise band.** Across ~30 arms tested, 3 exceeded P=0.80 where pure noise predicts ~5. Deep's 96% was confirmed on an independent higher-power grid; this is not.
* The effect is **0.6% of the gap to market**.

## Three parser bugs caught before they could corrupt the result

1. **Kickoff times leaked into team names** — `"12:30  Brighton & Hove Albion"`. An optional time group let the lazy name group absorb it, producing one "club" per kickoff time. That would have silently dropped a club's fixtures and made its rest days **wrong**, not missing.
2. **`AFC` is a prefix for Bournemouth** (`AFC Bournemouth`) and a suffix elsewhere (`Barrow AFC`).
3. **Middlesbrough was simply absent** from the alias map.

All 35 clubs now map; zero leaked times. Validated by asserting every club has fixtures.

## Reusable

`fixtures_raw.pkl` — 1,882 cup/European fixtures for all 35 EPL clubs, 2014-15 to 2025-26, with `aliases.py` for the name join. Any future idea needing a complete fixture calendar has it.

## Reproducing

```bash
python work_products/wp027_congestion/fetch_fixtures.py
python work_products/wp027_congestion/prepare_data.py
python work_products/wp027_congestion/premise.py          # residual-correlation check
python work_products/wp027_congestion/run_cv.py --workers 4   # 230 tasks, ~111 min
python work_products/wp027_congestion/analyse.py
```
