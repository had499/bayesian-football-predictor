# WP058 — market channel on the 343-window horizon-1 grid: REPLICATES but NOT SHIPPABLE

343/343 fits, 0 failed, 351 min. Arm: weight 0.5 (WP039's sweet spot) + WP051's tightened
shape prior (alpha 8, beta 2). Baseline is wp055's `H1_CHAMP` from the SAME data file and
grid, so the contrast is paired and within-protocol.

## Result

| | RPS |
|---|---|
| champion | 0.19657 |
| + market channel | 0.19579 |
| **dRPS** | **−0.00077 [−0.00186,+0.00027]** — CI includes 0 |

**The point estimate replicates WP039 almost exactly** (−0.00079 on the 115-window
TEST_LEN=3 grid), so the effect is real and protocol-independent. That was the question this
run existed to answer: the channel's effect size and the gap it is meant to close are now
measured on the same protocol.

## But three things block it

**1. LOSO consistency collapsed: 6/10 seasons, not 10/10.** WP039 reported 10/10 at the same
n (3,780). The effect is highly heterogeneous — 2016 −0.00480, 2021 −0.00380, 2023 −0.00350
are large wins; 2025 +0.00211 and 2022 +0.00155 are losses. A −0.00077 mean resting on three
good seasons is not the same claim as a consistent −0.00079.

**2. rhat is STILL 1.374** (median 1.063). WP051's tightened shape prior did NOT fix it
(WP039 was 1.41). The project's own standard: above ~1.1 is not reliably sampled and should
not ship whatever its RPS says. 0 divergences, so this is ill-conditioning, not a broken
model — the same root cause as the channel's 2x compute cost.

**3. It closes the gap by CONSUMING the gap's benchmark.**

| | gap to Pinnacle open |
|---|---|
| champion | +0.00286 [+0.00163,+0.00407] |
| + market channel | +0.00193 [+0.00088,+0.00300] |

33% of the gap closed — but by ingesting the opening price. This is legal at deployment (the
opening price IS available when predicting, which is the whole premise of the open-price
rule), so it is a genuine accuracy gain for anyone content to be market-dependent. It can
NEVER produce edge, and it makes every model-vs-market comparison circular. Consistent with
[[project-wp042-status]]: the channel reduced adverse selection only by pulling predictions
toward consensus.

## Verdict

**Do not ship.** Same conclusion as WP039, now established on the deployment-accurate
protocol, plus the new and less favourable information that LOSO consistency is 6/10 rather
than 10/10. The rhat is the hard blocker; the circularity is the reason it would not help
the goal even if the sampling were clean.
