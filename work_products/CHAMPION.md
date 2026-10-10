# The champion model

**As of WP021 (2026-09-29).** Encoded as `ModelConfig.champion()` in
`src/football_model/types/model_data.py`, guarded by `tests/test_champion_config.py`.

```python
from football_model.types.model_data import ModelConfig
config = ModelConfig.champion()
```

## What it is

`L0_baseline` + the **xG measurement channel** + the **deep-completion channel** + the **end-of-season stakes effort term**.

| Setting | Value | Why |
|---|---|---|
| `clip_theta` | **5.0** | harness value — NOT the dataclass default (2.0) |
| `center_team_strength` | **False** | harness value — NOT the dataclass default (True) |
| `use_xg_channel` | True, weight **0.75** | WP007, confirmed WP013, **weight tuned WP032** |
| `use_deep_channel` | True, weight 0.5 | WP020, re-tested WP021, **weight confirmed optimal and the channel finally established in WP033** |
| `use_stakes_effort` | True, sd 0.3 | WP028 — on expected value; best stability in the project |
| `season_start_window` | **3** (not the dataclass default 5) | **WP046 — on expected value.** −0.00008, better in **10/10** LOSO folds and **9/10** seasons |

**The two pins matter.** `scripts/run_cv_window.py` overrides those defaults, and every
cross-validated number in this project was produced with the overridden values. A plain
`ModelConfig()` is a *different model* from anything that has ever been measured here.
That trap is the main reason the config is now code rather than a line in a README.

## Evidence

| Channel | ΔRPS vs the arm below it | Status |
|---|---|---|
| xG channel | **−0.00258** [−0.00414, −0.00095] | **Established.** 6/6 leave-one-season-out folds (WP013) |
| xG weight 0.5 → 0.75 | **−0.00029** [−0.00059, +0.00002] | **Included on expected value.** Better in **10/10** leave-one-season-out folds, P(improves) ~97%, smooth interior peak on a 5-point curve (w0.25 is +0.00063 with a CI excluding zero; w1.75/w3.00 are catastrophic). 0.75 vs 1.0 is unresolved — what is established is that 0.5 was too low (WP032) |
| deep channel | −0.00083 [−0.00177, +0.00013] | **Now established (WP033).** Removing it from the current champion costs +0.00070, worse in **0/10** leave-one-season-out folds — it cleared the unanimous-fold bar for the first time, under the stress test of xG being upweighted to 0.75. Its weight of 0.5 is also a measured interior optimum (w1.0 is +0.00063, CI excluding zero) |
| stakes effort | −0.00014 [−0.00031, +0.00001] | **Not established.** Included on expected value, but negative in **10/10** leave-one-season-out folds, 7/10 seasons, coefficient sign-consistent 115/115, survives residualising against calendar and strength, and needs **no external data** |
| season_start_window 5 → 3 | −0.00008 [−0.00018, +0.00002] | **Not established.** Included on expected value: better in **10/10** LOSO folds and **9/10 seasons** — more season-consistent than either `deep` (6/10) or `stakes_effort` (7/10). P(improves) ≈ 94%, cost zero. Coherent with WP035, where `season_start_sigma_mult` was flat 2.0–3.0 and no-inflation scored +0.00004: all three say the model wants slightly LESS season-start flexibility. The effect is 1/12 of the MDE, so a tiny systematic CV-grid artefact would look identical — no mechanism is claimed (WP046) |

### Why `deep` is in despite a CI that includes zero

This is a deliberate departure from the project's usual bar, so the reasoning is recorded
rather than buried:

* **P(improves) = 95.9%**, E[improvement] = **+0.00083 RPS** ≈ 19% of the remaining gap
  to Pinnacle's pre-close line.
* The costs are asymmetric: a 96% chance of gaining 0.00087 against a 4% chance of losing
  0.00019. Demanding "CI excludes zero" means demanding 97.5% one-sided confidence —
  the right bar for *claiming a mechanism is real*, the wrong bar for *deciding what the
  model contains*.
* Even the worst subgroup — dropping 2016, its strongest season — is still **−0.00044**.
* It was **selected** on WP020's 30-window grid and **confirmed** on WP021's independent
  115-window grid, which added six seasons the selection never saw.

It is **not** claimed to be a real mechanism: better in only 6/10 seasons, and the effect
halves when one season is removed. Two separate questions, two separate answers.

**Updated by WP033.** The above was written when deep's only evidence was WP020/WP021.
WP033 removed the channel from the *current* champion and it cost +0.00070, worse in
**0/10** leave-one-season-out folds — so deep now clears the unanimous-fold bar on its
own, and no longer rests on the expected-value argument alone. It survived this while xG
was upweighted to 0.75, which was expected to make it redundant. The 6/10-seasons
weakness stands; the inclusion no longer needs special pleading.

### Why nothing else is in

Across 26 arms tested against the xG-channel baseline, 3 exceeded P(improves) = 0.80 where
pure noise across 26 tests predicts **5.2**. That cluster (`J_outcome_05` 85.9%,
`B_npxg` 81.7%, `D_shotxg` 77.4%) is indistinguishable from selection effects, and none has
been confirmed on an independent grid. Zero arms exceeded 0.90 on that grid.

Explicitly excluded despite being tested: the shot channel (WP019 null — and WP021 shows it
actively **cancels** deep, +0.00082), Dixon-Coles, outcome likelihood, split defence
(measurably worse), time-varying HFA/level, absence covariate, per-team sigma, per-season
kappa, form decomposition, multiplicative congestion (WP029), and **squad intactness
(WP030, measurably worse: +0.00090 [+0.00053,+0.00128], worse in 10/10 leave-one-season-out
folds)**.

WP030 is worth singling out because it is the one arm whose coefficient the latents did
*not* crush — 72% and 104% of prior SD, sign-consistent 114/115 windows, off the project's
strongest premise (r = +0.1469). It still degraded the forecast, because the thing it
corrected for (squad depletion) **autocorrelates**, so the champion's uncorrected latents
were already carrying a weak forecast of next-match availability. Removing that deleted
signal and put back nothing, since the XI is unobservable at the opening price. A
well-identified coefficient is not a better forecast.

## Data requirement

The champion now needs `ModelData.stakes_home/away` (WP028). Build them with
`work_products/wp028_stakes/stakes.py` + `prepare_data.py`; a plain `ModelData`
without them raises rather than silently dropping the term.

### Why the xG weight moved after six work packages at 0.5

0.5 was never measured — it was a hedge against double-counting, since goals and xG share
shot-level information. WP007 compared only 0.5 vs 1.0, on a 30-window grid whose MDE
(±0.0019) was four times the −0.00052 it measured, and called it null; WP013 then declined
to revisit it on the strength of that null. WP032 re-measured at 115 windows (MDE ±0.00095)
across five points and found a smooth interior peak at 0.75.

**The transferable lesson:** a null from an under-powered grid is not a closed question,
and citing one as if it were is how a free −0.00029 sat unclaimed for six work packages.

## Where it stands

**+0.0044 RPS behind Pinnacle's pre-close line** on the 115-window grid.

## Re-testing it

The high-power grid is ready: `work_products/wp021_power_confirmation/cv_shared_data_big.pkl`
(115 windows, 3,780 matches, deep + shots pre-joined) with a refitted `S3_xgchannel_only`
baseline in its `checkpoints/`. A new arm costs ~2.1 h wall at 4 workers.

Note when sizing a run: windows share ~97% of their training rows, so MDE improves more
slowly than 1/√n_windows suggests (WP021 predicted ±0.00080 and achieved ±0.00095).
