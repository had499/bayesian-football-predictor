# WP035 — Season-boundary and entry uncertainty

**Verdict: NULL, all four arms. No champion change. The one-window screen that
motivated this work package was pure noise, and that is the finding worth keeping.**

| arm | parameter | setting | ΔRPS vs champion | LOSO better |
|---|---|---|---|---|
| `SSM_1` | `season_start_sigma_mult` | 1.0 (none) | +0.00004 [−0.00023,+0.00032] | 1/10 |
| `SSM_2` | `season_start_sigma_mult` | 2.0 | −0.00001 [−0.00014,+0.00011] | 7/10 |
| `INIT_50` | `init_scale` | 0.5 | +0.00009 [−0.00031,+0.00050] | 2/10 |
| `INIT_100` | `init_scale` | 1.0 | +0.00030 [−0.00028,+0.00090] | 0/10 |

115 windows, 3,780 matches, 460 fits, 0 divergences, 225 min.

## The screen signal evaporated completely

                      screen (1 window)    actual (115 windows)
    SSM_1    mult 1.0      -0.00270             +0.00004
    INIT_50  init 0.5      -0.00261             +0.00009

Both were ~2.8x the MDE on one window, **and both had the opposite perturbation moving
the other way** (`mult 6.0` +0.00053, `init 0.05` +0.00175), which I argued made them
trustworthy in a way screen 1's ~1x MDE signals were not.

That argument was wrong. The signals were noise, and so was the apparent confirmation
from the opposite direction — with two draws from a noisy distribution, one on each side
of the champion, a spurious "gradient" is not unlikely.

**THE CALIBRATION, which is the real output of this work package.** Single-window ΔRPS
is reliable only at a *much* higher threshold than I assumed:

* **13-17x MDE** (WP032's xG weight 1.75/3.00 at +0.016/+0.013): reliable. Those arms
  were correctly killed without a sweep and the full curve confirmed the decision.
* **~2.8x MDE** (this work package): NOT reliable. Both leads reversed sign.
* **~1x MDE** (WP034's screen 1): not reliable, and the implied direction was wrong for
  the one parameter that moved.

So the smoke/screen is a good instrument for **which parameters respond** (the posterior
shift) and for **killing catastrophic settings**, and a bad one for anything smaller. Use
it to rank candidates and to rule out disasters; never to decide a direction.

## What the nulls mean substantively

`season_start_sigma_mult` = 3.0 and `init_scale` = 0.2 are both **confirmed adequate**.
`SSM_2` at −0.00001 and `SSM_1` at +0.00004 bracket the champion by nothing, which is
the signature of a flat region rather than a peak — the model is insensitive to how much
extra innovation variance it gets at a season boundary, anywhere in 2.0-3.0.

`init_scale` degrades monotonically as it grows (+0.00009 at 0.5, +0.00030 at 1.0, and
the screen's +0.00175 at 0.05 on the tight side), so 0.2 sits in a genuine interior
valley. Its influence on test rounds decays through the AR(1) anyway, which is the reason
I gave for expecting it to matter less than the multiplier — that part was right.

## Predictions on record vs outcome

* *"I expect at least one of these to clear the bar — this is the first time in the audit
  that a screen has shown a multi-MDE effect with a confirming opposite direction."*
  **Wrong.** Nothing cleared anything.
* *"My guess is SSM_1 or SSM_2 wins and init_scale turns out smaller than the screen
  implies."* Half right: `init_scale` was indeed the weaker of the two and for the stated
  reason (AR(1) decay), and `SSM_2` was nominally best — but at −0.00001, which is not a
  win in any meaningful sense.
* I noted before launching that I had been "wrong on direction in 4 of the last 5
  predictions". That is now 5 of 6.

## A real bug was found and fixed along the way

`ModelConfig.season_start_window` was **dead** — declared but never read, because
`prepare_model_data` used its own default of 5 and the harness never passed the config
value. The screen measured exactly +0.00000 for both 3 and 8, which is how it was
spotted: a genuinely inert parameter reads as exact zero.

**I then broke the harness fixing it.** The wiring referenced `config` above the line
that builds it, raising `UnboundLocalError` on every window; it killed 46 fits of a
running sweep, and because the next sweep was chained on "wait for 115 completed files"
— a condition a failure makes unreachable — nothing ran for 8h20m overnight. Three
separate errors: editing the harness while a sweep was using it, not checking the run's
exit status before building on it, and writing a test that grepped the source for a
string instead of executing the code, so it passed on broken wiring.

All fixed: the call now sits below `config`, and
`tests/test_champion_config.py::test_season_start_window_is_actually_wired` launches the
real harness with `season_start_window` set and asserts exit code 0.

    python work_products/wp035_season_boundary/run_cv.py --workers 6
    python work_products/wp035_season_boundary/analyse.py
