# Sweep performance — what works, what doesn't

Measured on the M5 Max (**6 "Super" + 12 "Performance" cores — all 18 are fast; there
are no efficiency cores**), 36 GB, JAX CPU backend, numpyro NUTS.

## The cost model

```
seconds ≈ 0.0024 × rounds²        (fixed overhead ≈ 0)
```

Quadratic because the latent array is `n_time × n_teams × 2` — about **29,000 latent
parameters** at the largest window (415 rounds × 35 teams × 2) — and NUTS runs **~53
gradient evaluations per draw**, ~258,000 per fit. A single logp+gradient is only
**1.68 ms**; the cost is the number of them.

Fixed overhead is ~zero, so batching several arms into one process saves nothing.

## THE STANDARD FOR NEW WORK: nutpie, tune=1000

```python
from arms import SAMPLER, sampler_args     # SAMPLER = nutpie, 600 draws / 1000 tune
cmd = [sys.executable, SCRIPT, ..., *sampler_args()]
```

**nutpie is ~2.4x faster with the same answer.** Measured over 230 real fits on the
115-window grid: **14.22 -> 5.89 CPU-hours**, and the arm effect is unchanged —

| sampler | deep vs no-deep |
|---|---|
| numpyro | -0.00083 [-0.00177, +0.00013] |
| nutpie | **-0.00086 [-0.00180, +0.00011]** |

Same model, same target posterior, same latent parameters. This is a faster
implementation of the same algorithm, **not** a precision trade.

**`tune` stays at 600** — matching `SAMPLER_LEGACY` and, crucially, matching the nutpie
baseline already fitted in `wp023_nutpie_baseline/checkpoints/`. `SAMPLER` therefore
differs from `SAMPLER_LEGACY` by **the backend and nothing else**, and that baseline is
reusable as-is, so a new sweep pays only for its own new arms.

**The accepted cost.** nutpie converges less cleanly than numpyro at this tuning:

| sampler | median rhat | p90 | max | >1.05 | >1.10 |
|---|---|---|---|---|---|
| numpyro | 1.013 | 1.030 | 1.055 | 3/230 | 0 |
| **nutpie** | **1.035** | **1.077** | **1.224** | **69/230** | **9** |

Divergences are 0 for both and the pooled arm effect is unchanged, but individual fits
are genuinely noisier — max per-window |ΔRPS| **0.00101**. That is tolerable *because
effects are read from 115 paired windows*, where independent per-window noise averages
out (it inflates the window-level SD of a paired difference by 0.4%, moving the MDE
0.00093 -> 0.00094). It would **not** be tolerable for a claim resting on one window.

Raising `tune` to 1000 was the alternative and would likely fix most of this, but it was
never benchmarked and would have invalidated the existing baseline. Revisit if
single-window rhat ever starts mattering.

### Two traps

1. **Never mix samplers inside one comparison.** The mean sampler shift is -0.00008,
   about **10% of the deep-channel effect**. It cancels only when both arms share a
   backend. Use `sampler_args()` rather than hand-writing flags — forgetting
   `--nuts-sampler` silently falls back to numpyro.
2. **The WP023 baseline is nutpie at tune=600, NOT the new standard.** It is matched to
   `SAMPLER_LEGACY`-era draws but not to `SAMPLER`'s tuning. Any new sweep must **refit
   its own baseline arm** at whatever settings it uses — which is the established pattern
   anyway (WP021 and WP023 both did).

Existing runners (WP007-WP023) are pinned to `SAMPLER_LEGACY` so they reproduce exactly.

## Earlier decision: full precision everywhere

**The user's decision (2026-09-29): keep the full posteriors and the full latent grid.**
Speed is not worth trading posterior quality for, given the effects being chased
(~0.0008 RPS) are close to the noise any shortcut introduces.

So `SAMPLER` (600/600, `target_accept` 0.9) and `time_grid_step=1` are the settings for
**every** run, and they are the defaults everywhere — `ModelConfig().time_grid_step` is
1, the CLI defaults to 1, and no sweep runner passes the flags.

`SAMPLER_FAST` and `--time-grid-step` remain available but are **opt-in and unused**.
They are kept because the measurements were worth recording, not because they are
recommended. Do not reach for them without asking.

| Purpose | draws/tune | grid step | relative cost |
|---|---|---|---|
| **Everything (the decision)** | `SAMPLER` (600/600) | 1 | 1.0x |
| *(available, not used)* screening | `SAMPLER_FAST` (300/300) | 2 | ~0.5x, costs \|ΔRPS\| up to 0.00042 |

Screening cost, measured against the 600/600 k=1 reference on two windows:

| window | 600/600 k=1 | 300/300 k=2 | speedup | ΔRPS |
|---|---|---|---|---|
| 70 | 132s | 72s | 1.82x | −0.00042 |
| 110 | 249s | 121s | 2.06x | −0.00014 |

Max per-window |ΔRPS| **0.00042**, i.e. 44% of the 115-window MDE (±0.00095), with 0
divergences and rhat unchanged. Both shifts are negative, so a systematic component
should largely cancel in a *paired* arm comparison — but that is an assumption, not a
measurement, which is why confirmation runs stay at the validated settings.

## Tested and REJECTED — do not re-test without a new reason

| Idea | Result | Why |
|---|---|---|
| `chain_method="vectorized"` | **~3x SLOWER** (1462s vs 509s) | See the section below — it is NOT the thread pinning, which was my first (wrong) guess |
| `target_accept` 0.9 → 0.85 | Slower **and** worse | ESS fell 790 → 590 for no time saving |
| Closed-form AR(1) replacing `pytensor.scan` | **0.70x** | `rho^(-c)` over a 414×35 matrix costs more than `lax.scan` saves; compile 0.2s → 7.6s. Gradients agree to 9e-13, so the derivation was right — it is simply slower |
| Batching arms per process | No gain available | Fixed overhead measured at ~0s |
| More/fewer workers | Already optimal | 4 workers × 4 chains = 16 of 18 cores |
| `--time-grid-step 3` | **Unsafe** | ΔRPS 0.00130 — larger than the deep-channel effect (0.00083) it would be used to measure |

## The biggest lever is not a setting

**Stop a sweep once the ordering is decided.** WP022 was halted at 88/115 windows because
the monotone pattern and two decisively-worse arms had already settled it — that alone
saved more time than any tuning here (23% of that sweep). A partial sweep is a valid
paired comparison provided every arm covers the same windows.

Second: don't run confirmation-grade precision for a screening question.

## A latent bug this work exposed

`t_idx` was doing two jobs — indexing the latent walk **and** labelling which round a
match is in. Test-row selection (`run_cv_window.py`) and the odds join both key on it,
so coarsening the grid silently mislabelled rounds. It happened to crash; it could just
as easily have returned corrupted results.

The two are now separate: **`t_idx` is the true round** (identification only) and
**`t_walk` indexes the latent walk**. `tests/test_wp023_time_grid.py` pins them apart
and asserts `k=1` is bit-identical to every pre-WP023 result.


## Why `vectorized` is ~3x slower — measured, not guessed

My first explanation was that pinning `intra_op_parallelism_threads=1` and disabling
multi-threaded Eigen left the vmapped kernel stuck on one core. **That is wrong.**
Unpinning the threads recovers only 12%:

| chain_method | XLA threads | secs | vs best |
|---|---|---|---|
| parallel | pinned | **86s** | 1.00x |
| parallel | unpinned | 86s | 1.01x |
| vectorized | pinned | 263s | 3.06x |
| vectorized | unpinned | 231s | 2.69x |

(Window 70, draws/tune 200. Note `parallel` is identical either way, so those XLA thread
flags do nothing for the mode we actually use.)

The real reasons are structural, and all three point the same way:

1. **The batch dimension is 4, in float64.** ARM NEON is 128-bit, so float64 SIMD is
   **2 lanes**. Batching 4 chains fills two vector registers — nearly no SIMD gain —
   while still paying 4x the arithmetic and 4x the memory traffic. On CPU this is close
   to "just do 4x the work".
2. **The inner structure is a 415-step sequential `lax.scan` over 35 floats.** vmap
   cannot shorten the 415 steps, and it cannot amortise per-step overhead because the
   ops are tiny either way (35 floats -> 140). Vectorisation pays when ops are large
   enough to hide overhead; this is the opposite shape.
3. **NUTS trajectories must run in lockstep.** Chains build trees of different depths.
   Under `vmap` every lane executes together, so all chains run to the DEEPEST chain's
   tree depth and the shorter ones do wasted work. Under `pmap` each chain stops at its
   own depth.

`pmap` across devices — what `chain_method="parallel"` already does — is the right tool
for 4 independent chains of small sequential work.

---

## GPUs: NO, and the reason is structural (2026-10-10)

Asked whether a GPU could substitute for cores. It cannot, and the two measurements above
already explain why rather than this being a guess:

- `chain_method="vectorized"` -- the standard way to use a GPU for MCMC -- is **3x SLOWER**.
- The closed-form AR(1) that would turn the scan into one GPU-shaped matrix op was
  implemented, verified correct (gradients agree to 9e-13), and is **0.70x**.

**CORRECTION (2026-10-10, prompted by the user pushing back twice).** My first answer
cited the two measurements above as proof that GPUs cannot help. That reasoning was INVALID:
both were measured on CPU, and both reject exactly what a GPU would need. `vectorized` is
slower on CPU because you already have cores there; on a GPU it is the ONLY source of
parallelism. The closed-form AR(1) is slower on CPU because a 414x414 matmul is expensive
there -- and its recorded result, "more arithmetic, gradients correct to 9e-13", is precisely
the *more FLOPs, less sequencing* trade that GPUs are built for. Neither number transfers.

The correct hardware-independent argument is narrower: *the inner structure is a 415-step
sequential `lax.scan` over 35 floats*. A GPU needs thousands-to-millions of parallel
elements per op; 35 floats is nothing. A GPU run would launch 415 sequential kernels per
gradient evaluation, each doing microseconds of work, and drown in launch overhead. This
model is **latency-bound on a tiny sequential recursion** -- the single worst shape for a
GPU. No amount of GPU money fixes that; it is a property of the model, not the hardware.

### What to do instead on a tight compute budget

1. **Screen on the CLOSE-LINE metric, not outcome-RPS, and cut windows WITHIN a protocol
   -- never by switching protocol.** WP057 measured the close metric at 4-5x the statistical
   power, so the same resolution needs 4-5x fewer TEST MATCHES.

   **CORRECTION (user caught this immediately).** I first wrote "use the ~115-window grid
   instead of the 343-window one". That is WRONG, and it is a trap worth spelling out. The
   two grids are different PROTOCOLS, not different sample sizes: 343 windows exists because
   TEST_LEN=1 (refit every round), 115 because TEST_LEN=3 (refit every three). Measured on
   the IDENTICAL 3,780 fixtures both contain, TEST_LEN=1 scores 0.19657 against TEST_LEN=3's
   0.19692 -- **paired dRPS -0.00035**. The cheap grid measures a STALER, WORSE model.
   Compounding it, effect sizes do not transfer across protocols either: the GBM wrapper is
   -0.00063 on TEST_LEN=3 but -0.00040 at horizon 1, so the cheap grid misleads on MAGNITUDE
   as well as on baseline.

   The valid saving KEEPS the protocol and uses fewer of its windows -- e.g. TEST_LEN=1 on
   every OTHER window, 170 of 343, which halves compute while every fit remains a genuine
   single-round refit. That loses only precision, which is exactly what the close metric
   buys back. **Cutting windows within a protocol is free; switching protocol is not.**
2. **Split a sweep across machines by window range.** `run_cv.py` caches per-window
   checkpoints and skips existing ones, so two machines can share one sweep with no
   coordination beyond copying checkpoints back.
3. **Enable the C backend** -- `PYTENSOR_FLAGS="cxx="` costs ~15% (145s -> 124s measured on
   a real 600/600 nutpie fit) and the breakage it worked around is fixed.
4. **Spot/preemptible instances** if cloud is ever used: this workload is perfectly
   interruptible because every window checkpoints independently.

### The two routes that COULD work, and why they still do not here

1. **Closed-form AR(1)** -- `x = L @ (sigma*z)` with `L[t,s] = rho^(t-s)`, removing the
   sequential dependency entirely. The masking features are solvable, contrary to my first
   claim: `season_start_sigma_mult` folds into `z`, `entry_shift` is a second triangular
   product, and `active_mask` becomes `exp(log(rho) * C)` with `C` a PRECOMPUTED integer
   matrix of active-step counts per team (35 x 416 x 416 ~ 24MB).
2. **vmap across WINDOWS** -- 343 independent fits, padded and masked, give ~1,372 chains in
   one pass. This is the large multiplier, and CPU parallelism caps at ~7x.

**Why it dies on the available hardware (GTX 1050 Ti).** We sample in **float64**
(`pytensor.config.floatX = float64`, nutpie is double precision). Pascal FP64 runs at 1/32
rate: **~66 GFLOPS**, LESS than the 18-core CPU delivers. And 4GB VRAM caps route 2 at
~50-100 chains against the ~1,372 that would make it worthwhile.

float32 is not the escape hatch: rhat already sits at 1.02-1.21 (the market channel hit
1.37), so single-precision leapfrog on a 29,000-dim ill-conditioned posterior means more
divergences and worse rhat -- and it contradicts the standing keep-full-precision decision.

**GENERAL RULE: every consumer NVIDIA card cripples FP64** (1/32 Pascal, 1/64 Ampere/Ada),
so a 3090 or 4090 would not fix this either. Usable FP64 means A100/H100, i.e. cloud.
**Conclusion: the lever is fewer and cheaper experiments, not more compute.**
