# WP038 — State-free xG, and the shot-level direction

**Verdict: DEAD ON PREMISE. ~15 minutes, no CV sweeps. And the premise test produced a
quantitative CALIBRATION ANCHOR that presumes the whole shot-level direction dead.**

## The idea and why it was the best remaining candidate

WP037 closed the last structural lead and established that across ~25 arms **only
LIKELIHOOD changes have ever improved this model** — new observations about the same
matches. The xG channel (−0.00258) is the only established win and WP032 showed its
weight was under-set, so the model demonstrably values xG as a second measurement.

I had claimed we therefore needed new DATA. That was wrong: the Understat **shot-level**
feed was already cached on disk (12 seasons, 4,560 matches, every shot with
`xG / situation / result / minute / X / Y`), used only by WP014 to strip penalties. The
*match-aggregate* feed is exhausted; the shot-level one was barely touched.

State-free xG was the pick because its mechanism was independently established: WP018
proved score-state dependence is real, so a side 2-0 up stops attacking and a side
chasing inflates its xG. Measuring xG only while the score was LEVEL should be a cleaner
read on the same latent.

## It failed twice

**As a substitute** — rolling means over a team's previous K matches, predicting its
next-match goals:

      K   goals(base)     full xG    level xG  1st-half xG
      5        0.1983      0.2284      0.1601      0.1829
     10        0.2362      0.2622      0.2088      0.2255
     20        0.2667      0.2825      0.2377      0.2540

Full xG wins at every K. A plain bias-variance trade: level-state is only 45% of xG and
averages 44 minutes a match (p10 = 9 minutes), so the contamination removed is worth
less than the exposure lost.

**As a complement** — leave-one-season-out OOS R² (a nested in-sample gain is vacuous,
BACKLOG caution 7):

    full xG only     +0.06864
    full + level     +0.06841      <- no gain
    full + goals     +0.07034      <- the control DOES gain, so the test has power

## The calibration anchor — the real output

Testing every shot-level candidate the same way:

    full + raw shot count          +0.00484   <-- WP019 BUILT THIS. It was NULL.
    full + setpiece                +0.00313
    full + openplay/setpiece split +0.00217
    full + shots in box            +0.00190
    full + goals  [CONTROL]        +0.00170
    full + big chances             +0.00136
    full + level xG                -0.00023

**Raw shot count is the largest single addition on this scale, and WP019 already built
it as a measurement channel: null, and it actively CANCELLED the deep channel
(+0.00082 worse).** So a predictor can add this much OOS R² for next-match goals and
still fail as a channel.

That gives a threshold. **Every shot-level candidate sits BELOW the known-failure point**,
so all of them are presumed null and the direction is closed without a single sweep.
This also quantifies BACKLOG caution 13 ("the per-match premise check over-promises —
treat it as an upper bound", previously 3-for-3): the upper bound is now anchored to a
measured failure rather than being a vague warning.

Why a good predictor can fail as a channel: a measurement channel has to inform the
LATENT. Shot volume is partly a style fact (a team that shoots a lot from distance) rather
than a sharper read on the scoring rate, and it conflicts with what xG already tells the
same latent — which is exactly what WP021 measured when shots cancelled deep.

## Process note

The first run of this premise joined the shot cache to the panel on `match_id` and kept
**124 of 9,120 rows** — the panel's `match_id` is an internal `int16` index (0, 0, 5, …),
not the Understat id. Every correlation in that run was garbage (the −0.83 in the
robustness table gave it away). Fixed by joining on `(team, date)`, the key WP030
established as unique, with an assertion that the join retains >90% of rows. This is the
fourth appearance of the join-key trap in this project; the assertion is now in the
script so it cannot pass silently again.

    python work_products/wp038_state_free_xg/premise.py    # substitute + complement
    python work_products/wp038_state_free_xg/premise3.py   # all shot-level candidates
