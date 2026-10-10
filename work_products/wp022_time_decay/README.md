# WP022 — Dixon-Coles time decay

**Status: complete (stopped early at 88/115 windows — the answer was already decisive). Verdict: WORSE, not merely null.** Time-decay weighting degrades the champion monotonically. Two of the three arms are decisively worse with CIs excluding zero; the third, the weakest decay, is indistinguishable from no decay at all.

**The champion config is unchanged.** `use_time_decay` stays `False`.

## What this tested

The **other half of Dixon & Coles (1997)** — the half this project never implemented. WP002/WP007 tested only the tau correction. Decay weights each observation by `exp(-xi * age)`, which is a genuinely different mechanism from the AR(1) walk: a walk lets team strengths **drift**, decay changes how much each observation **counts**.

Applied to the whole likelihood — goals *and* both measurement channels. If only goals decayed, old matches would still count fully through xG and deep, and the arm would measure a diluted version of itself.

## Results

88 windows, 2,918 paired matches, against WP021's `H_deep_added` (which *is* the champion, so the baseline was free).

| Arm | half-life | Pooled RPS | vs champion | |
|---|---|---|---|---|
| champion | no decay | **0.19633** | — | |
| `N_decay_810` | 810 days | 0.19651 | +0.00019 [−0.00050, +0.00089] | no difference |
| `N_decay_270` | 270 days | 0.20103 | **+0.00471** [+0.00267, +0.00690] | **worse** |
| `N_decay_090` | 90 days | 0.21028 | **+0.01395** [+0.01015, +0.01768] | **much worse** |

Monotone: the more the model decays its history, the worse it forecasts. The optimum is at no decay or weaker.

## Why — measured before running, not rationalised after

Normalising the weights preserves total likelihood **mass** but not **information**. On a real training window (rounds 1–253, n=2,780, spanning 7.3 years):

| half-life | effective N | share of data retained |
|---|---|---|
| 90 days | 250 | **9%** |
| 270 days | 800 | **29%** |
| 810 days | 1,970 | 71% |
| no decay | 2,780 | 100% |

The RPS ordering is exactly the effective-N ordering. Decay here buys nothing and costs sample size, because **the walk already extracts the time structure**. Dixon & Coles needed decay because their strengths were static; ours are not.

The sampler agrees:

| Arm | divergences | max rhat |
|---|---|---|
| `N_decay_090` | **5** | **1.755** |
| `N_decay_270` | 0 | 1.108 |
| `N_decay_810` | 0 | 1.053 |

Only the arm that discards 91% of its effective data fails to converge — the posterior is genuinely under-identified, which is the same fact showing up a second way.

## Two predictions scored

* **The premise check was WRONG.** It put the optimum at a 270-day half-life (−1.01% MSE on a decay-weighted mean of past xG, 9,120 team-matches). The model's optimum is at no decay. The premise optimised a *static weighted mean with no latent walk to lose precision in*, so its tradeoff was not the model's tradeoff.
* **The pre-run update was CORRECT.** After measuring effective N I revised to "810 best of the three, plausibly all worse than no decay", and recorded it in `run_cv.py` before launching. That is what happened.

This is the **third consecutive time** the per-match premise-check framework has over-promised (WP019 shots, WP020 deep, now decay). It is reliable for ruling candidates OUT via redundancy and unreliable for predicting gains. Treat its point estimates as upper bounds.

## Stopped early, deliberately

Halted at 88/115 windows. The 56-window partial already showed the monotone pattern and two decisively-worse arms; the remaining 27 windows would have bought precision on a settled question at ~80 minutes of compute. Recorded because "we ran it to completion" is not a virtue when the conclusion is fixed — and the partial read is a valid paired comparison, since all three arms cover the same 88 windows.

## Reproducing

```bash
python work_products/wp022_time_decay/run_cv.py --workers 4   # 345 tasks; stop once the ordering is clear
# analysis is inline — see the arms vs wp021_power_confirmation/checkpoints/H_deep_added
```
