# WP060 — the two confirmed post-hoc gains COMPOSE (93-97% additive)

Both corrections are post-hoc (neither subject to the latent-crushing failure mode) and
both were confirmed on the closing-line metric, but had never been applied together.
Coefficients were fixed on windows <=200 for BOTH (GBM config on 70-200, injury k on
<=200), so windows 201-343 is genuinely held out and nothing here is tuned on the
evaluation data. n=1,388 matches with a GBM correction, availability and a Pinnacle close.

## Result

| arm | RPS | dRPS vs OUTCOME | dRPS vs CLOSE | LEVEL |
|---|---|---|---|---|
| baseline (champion) | 0.19636 | — | — | 0.000043 |
| GBM wrapper only | 0.19603 | −0.00033 [−0.00081,+0.00017] | **−0.000095 [−0.000159,−0.000032]** | 0.000045 |
| injuries only | 0.19614 | −0.00022 [−0.00059,+0.00014] | **−0.000103 [−0.000154,−0.000052]** | 0.000040 |
| **BOTH** | **0.19583** | −0.00053 [−0.00119,+0.00012] | **−0.000185 [−0.000273,−0.000098]** | 0.000042 |

## They are nearly INDEPENDENT

| | sum if independent | BOTH actually | captured |
|---|---|---|---|
| CLOSE | −0.000198 | −0.000185 | **93%** |
| OUTCOME | −0.000543 | −0.000528 | **97%** |

And each adds over the other, both CIs excluding zero on the close:
- BOTH vs GBM only: **−0.000090 [−0.000140,−0.000039]** — injuries add over the GBM
- BOTH vs injuries only: **−0.000082 [−0.000144,−0.000019]** — the GBM adds over injuries

So the overlap is ~7%. The worry that the GBM's 253 form features already reflect absent
players through recent output is **not** borne out: published injury data carries information
the form features do not.

The WP057c guard passes: LEVEL is 0.000043 at baseline and 0.000042 with both corrections —
it does NOT grow, so neither gain is drift toward the market's average.

## Effect on the gap to Pinnacle's opening line (same held-out subset)

Pinnacle open scores 0.19337 here.

| model | RPS | gap to open |
|---|---|---|
| champion | 0.19636 | +0.00299 [+0.00102,+0.00496] |
| + GBM wrapper | 0.19603 | +0.00266 [+0.00073,+0.00467] |
| + injuries | 0.19614 | +0.00277 [+0.00084,+0.00466] |
| **+ BOTH** | **0.19583** | **+0.00246 [+0.00051,+0.00441]** |

**~18% of the gap closed** by two corrections that cost no MCMC.

## Caveats, stated plainly

- **Not established on outcome-RPS.** Every outcome-RPS CI here includes zero, including the
  combination against baseline (−0.00053 [−0.00119,+0.00012]). The result rests on the
  closing-line metric, which WP057 validated at 4-5x power and whose LEVEL/DISCRIM guard
  passes. The project standard already rejects CI-excludes-zero as the inclusion bar, but
  this is a weaker claim than an outcome-RPS confirmation would be.
- **The gap improvement is not independently significant** — the gap CIs overlap heavily
  across all four rows. What is established is the PAIRED improvement, not a new gap level.
- **The injury half carries an external Transfermarkt dependency**, the same objection that
  kept the Elo anchor out of the champion. An operational decision.
- n=1,388 and availability is present on both sides for 94.0% of it; the rest falls back to
  no adjustment.
- Many close-metric tests have now been run this session. These two increments both exclude
  zero AND the composition is internally coherent (93-97% additive, LEVEL flat), which is a
  consistent picture rather than one lucky cell — but the multiplicity is real.
