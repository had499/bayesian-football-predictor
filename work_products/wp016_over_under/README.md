# WP016 — Over/under 2.5: does the model beat, or add to, Pinnacle on totals?

**Status: complete. Neither pre-declared test hit: the model adds nothing to Pinnacle's over/under 2.5 price either (Q1 goes the wrong way, Q2 gets random-quality prices). One exploratory, model-free result is worth a follow-up: the best available over/under price against Pinnacle's fair price. See "Results".**

## Why

WP012 and WP015 found the model adds nothing to Pinnacle's 1X2 price. But 1X2 is a lossy summary of what the model actually produces, a distribution over scorelines. The over/under 2.5 market prices that same distribution directly, and it was never tested. It is the last market whose odds are already in `odds_raw.pkl`.

## What was built

- **`dc_over_prob(lambda_home, lambda_away, rho, line)`** in `football_model.model.predict`: P(total goals > line) from the same Dixon-Coles scoreline grid `dc_outcome_probs` uses. To guarantee both markets come from one distribution, the grid construction was pulled out into a shared `_dc_score_grid`; `dc_outcome_probs` behaves exactly as before (its existing tests still pass).
- **5 new tests** (`tests/test_predict.py`): agreement with the closed-form Poisson sum for independent goals (1e-9), a brute-force scoreline loop with the Dixon-Coles correction (1e-12), `rho=0` equals `None`, monotonicity in the goal rate and the line, and that 1X2 is unchanged.
- **`wp016_over_under.ipynb`**, reusing `football_model.evaluation.market`.

## Data and sample

- Odds: Pinnacle over/under 2.5, pre-closing (`P>2.5`, `P<2.5`) and closing (`PC>2.5`, `PC<2.5`); about 92% of matches. Result: `FTHG + FTAG > 2.5`. Every joined score is checked against the odds file.
- Model: `baseline` predictions from **both** held-out sets: WP001's 401 matches (predicting 1 round ahead) and WP014's 1,086 (1–3 rounds ahead). Over/under was never looked at on either, so neither set is contaminated by selection on this question, and the primary test **pools them** (about 1,360 matches with Pinnacle over/under odds). Old and new are also reported separately.
- Model: `baseline`, the same one WP012/015 tested. The best arm is exploratory.

## Pre-declared primary tests

| # | test | a "hit" requires |
|---|---|---|
| Q1 | leave-one-window-out binary log loss, Pinnacle-close logit + model logit minus Pinnacle-close logit alone (per match); negative = model helps | 95% CI entirely below zero and the same sign in both halves (split at the median date) |
| Q2 | mean CLV per bet of model-selected bets at Pinnacle's **opening** over/under odds, `tau = 0.02`, against Pinnacle's closing fair prices (two-way de-vig) | 95% CI entirely above zero and the same sign in both halves |

Both same rules as WP012's P1 and P2. Everything else is exploratory: separate old/new results, the blend sweep, the `tau` sweep, the model's total-goals bias, the best arm, and a model-free soft-vs-sharp table (Bet365 and best-price closing odds against Pinnacle's fair price).

## What each outcome means

- **Neither hits.** No edge from this model on 1X2 or on totals. Stop the "beat the bookmakers with this model" line for everything the data covers; what remains is information the market lacks, or markets with softer prices.
- **Q1 or Q2 hits.** The goal-distribution model carries information about totals that Pinnacle does not price. A lead, not proof: it would need a forward test on live odds before any staking. The Asian handicap market would then be the natural next check.

## Results

1,355 held-out matches with Pinnacle over/under odds (360 from WP001's set, 995 from WP014's), model = `baseline`. The notebook was run once. The actual over-2.5 rate is 56.5%; Pinnacle's closing mean P(over) is 55.2%; the model's is 52.5%. Total goals per match: actual 2.95, model 2.80. So **the model under-predicts goals by about 5%**, in line with WP010's finding that its goal rate runs low.

| # | test | n | estimate | 95% CI | half 1 / half 2 | result |
|---|---|---|---|---|---|---|
| Q1 | log loss, Pinnacle + model − Pinnacle alone (negative = model helps) | 1,355 | **+0.00097** | [+0.00059, +0.00136] | +0.00123 / +0.00072 | **no hit** (wrong side of zero) |
| Q2 | CLV per bet at Pinnacle's opening odds, `tau=0.02` | 931 bets | **−3.0%** | [−3.4%, −2.6%] | −2.8% / −3.2% | **no hit** |

**Reading.**
- **Q1:** the model does not help, and adding it makes log loss slightly worse, +0.001. Read that as the cost of fitting an extra coefficient that carries no information, not as the model being actively misleading. It is on the unfavourable side in both halves, in both held-out sets (old +0.0018, new +0.0013), and for the best arm (+0.0015, CI [+0.0009, +0.0021]).
- **Blends never help:** every weight has a worse point estimate (+0.0001 at `w=0.05`, +0.0009 at 0.2, +0.0037 at 0.5, +0.0126 at 1.0).
- **Q2:** model-selected bets get −3.0% CLV against −3.2% for betting every over and under (the control, about Pinnacle's margin), and it does not improve as the threshold rises (−2.9% at 0.05). The best arm is the same (−2.7%). The model has no information about where the over/under line closes.
- Old and new sets agree on both tests (Q2: −2.5% and −3.2%).

**Verdict by the declared rule:** no hit on either test, so the model has no edge on totals. Together with WP012 and WP015 (1X2), it has none on either market that the data covers. Asian handicap prices the same goal distribution, so I would not expect a different answer and do not recommend building its settlement logic on this evidence.

**Exploratory, model-free: best available closing price vs Pinnacle's fair price (2,099 matches).** No model is involved. Bet the best price (`Max`) whenever it beats Pinnacle's de-vigged closing probability by more than `tau`:

| `tau` | bets | claimed edge | realised ROI | 95% CI |
|---|---|---|---|---|
| all outcomes (control) | 4,198 | +0.1% | +0.1% | [−0.8%, +1.0%] |
| 0.02 | 831 | +4.4% | +3.7% | [−4.0%, +11.4%] |
| 0.03 | 528 | +5.6% | +8.4% | [−1.9%, +18.5%] |
| 0.05 | 212 | +8.1% | +13.3% | [−3.8%, +30.6%] |

Realised ROI tracks the claimed edge and rises with `tau`, which is what real value would look like, but every CI includes zero, and this is one of several soft-vs-sharp tables looked at across markets (WP012's 1X2 version had a claimed +6.2% at `tau=0.02` and realised −0.9%). Bet365's own prices qualify on too few bets to say anything (17–75 bets, CIs spanning ±50% or more).

**Treat it as a lead, not a finding.** Reasons for caution: `Max` is the single best price across many books at the close, so it may include stale, off-market or stake-limited quotes that could not actually be bet; it is an in-sample upper bound; and the sign pattern rests on 200–800 bets. If it is worth pursuing, it needs new data, and it does not need the model at all: football-data.co.uk publishes the same closing over/under columns for the other top leagues (not checked to carry Pinnacle and best-price columns for all of them), which would give a fresh, larger sample.

## What this cannot show

- Realised profit (no stake limits, restrictions or liquidity).
- Asian handicap: over half its closing lines are quarter lines (stake split, possible push), which need their own settlement logic. Deferred until this test says whether it is worth building.
- Anything beyond EPL, 2020–2026.

## Reproducing

```bash
pytest tests/test_predict.py
cd work_products/wp016_over_under
jupyter lab wp016_over_under.ipynb   # no sampling, about a minute
```
