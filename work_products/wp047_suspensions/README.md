# WP047 — Suspensions: the clean absence signal, and it carries nothing

**Verdict: DEAD ON PREMISE, well-powered. corr = +0.0043 [−0.0302, +0.0396], incremental
OOS R² −0.00013 against a +0.005 bar. This closes the absence direction properly.**

## Why this was not just the fifth absence test

WP011 (lineups), WP012 (absence covariate), WP016 (keepers) and WP030 (squad intactness)
all failed. I argued those nulls were weaker than they looked, because every one measured
availability from MINUTES PLAYED — which conflates long-term injury, matchday fitness
calls and manager rotation. Two of those three are not knowable when the opening price is
posted, so none of those tests could have explained our deficit TO THE OPEN.

A red card differs on every axis that matters: **exogenous** (not a manager's choice, not
a fitness judgement), **pre-open knowable** (public the moment it happens, days before the
next price), and **deterministic** (the ban is automatic). It is the only absence that is
certain and public at open time.

## Result

    suspension flag resolved for 89.3% of the 3,780 test matches
    home side suspended 6.0% | away 5.3% | either 11.1% (n=376)

    corr(susp_diff, tilt) = +0.0043 [-0.0302,+0.0396]
      [scale: intactness +0.1469, Elo +0.047, stakes +0.040, congestion -0.020]

    home side suspended   n= 204  tilt -0.0716 [-0.3118,+0.1873]
    away side suspended   n= 179  tilt -0.0305 [-0.2920,+0.2479]
    neither               n=3000  tilt -0.0350 [-0.1001,+0.0321]

    incremental OOS R^2 for next-match goals:  -0.00013   (bar +0.005)

**Neither hypothesis holds.** (a) "the side is a player short so it underperforms" predicts
home-suspended NEGATIVE and away-suspended POSITIVE; both come out mildly negative.
(b) "the model over-reacted to a 10-man result and now under-rates them" predicts the
opposite signs; also absent. The incremental OOS R² is NEGATIVE, 40x below the level at
which WP019's shot count already failed as a channel (+0.00484).

**Power:** the correlation CI half-width is ±0.035, so this detects an effect the size of
anything that has ever worked in this project (stakes +0.040, Elo +0.047). A real null.

## What it settles

My caveat on the four earlier absence nulls — that they used crude proxies and could not
speak to the open-price deficit — **is now answered, and they were right.** The cleanest,
most exogenous, most certainly-pre-open absence signal available carries no information
the model lacks. The absence direction is closed on five independent tests.

## Limitation

football-data gives TEAM-level cards, so we know a team will be a player short but not
WHICH player. Yellow-card accumulation (5 → 1 match ban) needs player-level data we do
not have. So a player-weighted version of this test remains untested — but given a flat
result on the binary flag, and four prior nulls, the prior on it is poor.

    python work_products/wp047_suspensions/premise.py
