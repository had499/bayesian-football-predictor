# WP016 — Splitting defence into chance suppression and shot-stopping

**Status: complete. Verdict: measurably WORSE, not null.** Splitting the defence latent costs **+0.00027 RPS [+0.00005, +0.00049]** — the CI excludes zero on the wrong side. This is the first change tested in this project that actively degraded the model rather than doing nothing.

**Do not adopt. The shipping config remains `L0_baseline` + `use_xg_channel=True` (WP013).**

## Why it looked promising

Everything proposed before this was attacking-side. The model always *had* a defence latent, but nothing observed defence separately — and defence turns out to be two skills with very different repeatability:

| | cross-team SD | season→season persistence |
|---|---|---|
| Chance suppression (xG faced) | 0.333 | **r = +0.65** |
| Shot-stopping (xG faced − goals conceded) | 0.180 | **r = +0.22** |

One latent with one `sigma` was learning a strongly-repeatable skill and a barely-repeatable one at the same rate. That's a genuine structural mismatch, and unlike WP015's home-advantage claim these persistences are directly measured rather than inferred from a noisy range.

## The design

Two defensive latents with separate innovation SDs. The identification is the interesting part: **the xG channel sees suppression only**, while goals see suppression *and* stopping — so the gap between goals conceded and xG conceded is what pins shot-stopping down. Feed both channels the same rate and the two latents collapse. There is a test asserting they don't, and `build_model` refuses `split_defence` without the xG channel.

`defence` is still registered as suppression + stopping, so `predict.py`, the CV harness and the service are untouched.

Two arms, because the tight `sigma` prior on stopping is itself a modelling choice:

| Arm | `stop_sigma_scale` |
|---|---|
| `H_split` | 0.3 (tight — only ~a fifth of save variation persists) |
| `H_split_loose` | 1.0 (stopping as free as suppression) |

## Results

| Arm | Pooled RPS | Δ vs shipping config | CI (cluster) |
|---|---|---|---|
| `S3_xgchannel_only` | 0.20481 | — | — |
| `H_split` | 0.20508 | **+0.00027** | **[+0.00005, +0.00049] — worse** |
| `H_split_loose` | 0.20511 | +0.00030 | [+0.00007, +0.00055] — worse |
| sigma prior (loose vs tight) | | +0.00004 | [−0.00001, +0.00008] — no difference |

Leave-one-season-out: positive in all six folds, "worse" in four of six. Gap to Pinnacle pre-close widens slightly, +0.00409 → +0.00433. Sampler clean: 0 divergences, max rhat 1.048.

**The sigma prior is not the culprit.** Tight and loose versions are indistinguishable from each other, so this isn't a case of over-shrinking the new latent.

## Why it likely hurt — but the mechanism is NOT diagnosed

The plausible story: the split adds roughly 14,600 latents (417 rounds × 35 teams) estimated from the same data. Predictions use the *combined* defence, so if the second latent is mostly noise, that noise flows straight into the prediction. The persistence asymmetry that motivated the split cuts both ways — stopping persisting at only r=+0.22 means most of what the new latent can express *is* noise, and giving noise its own parameter lets it into the forecast.

**But I cannot confirm that**, because of a reporting gap: `sigma_stop` was never added to the harness's recorded parameters, so there is no record of what the model actually learned for it. I can't tell whether it learned a distinct stopping latent or effectively collapsed it. `run_cv_window.py` now records it, but re-running costs ~12 hours and the RPS answer wouldn't change, so this is left undiagnosed.

## Two process notes worth recording

1. **I underestimated the compute by 14×.** Predicted ~50 minutes; it took **702 minutes**. Splitting defence adds ~50% more latent parameters and per-window time rose from ~190s to ~585s as training sets grew. Estimate sweep cost from *parameter count*, not from a previous sweep's wall-clock.
2. **Capture the new parameter in `scalar_names` when adding one.** This is the second time a diagnostic was unavailable after the fact (WP014's per-season `kappa` had the same problem). Cheap to do at build time, expensive to recover.

## What this means for the goalkeeper follow-up

The natural next step was to tie the shot-stopping latent to keeper identity — the one position where an individual's absence is both high-impact and knowable in advance. **I would not do that now.** It builds on a split that measurably degrades the model, so the foundation is wrong. Any keeper work would need the split to at least be neutral first.

## The blind spot this exercise did expose

Worth keeping separately from the negative result, because it stands on its own: **WP011 and WP012's absence nulls never tested goalkeepers.** The `absent_drop` feature weighted players by xG+xA, and keepers average **0.003** against **0.193** for outfielders — so a missing first-choice keeper registered as ~zero absence. Keeper changes occur in **9.4%** of matches. Those nulls are correctly scoped as "outfield absence doesn't help", not "absence doesn't help".

## Reproducing

```bash
python work_products/wp016_split_defence/run_cv.py --workers 4   # 60 tasks, ~12 HOURS
python work_products/wp016_split_defence/analyse.py
```

Needs WP007's `cv_shared_data.pkl`/`checkpoints/`, WP013's `checkpoints/`, WP008's `odds_raw.pkl`.
