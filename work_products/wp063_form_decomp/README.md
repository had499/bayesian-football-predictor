# WP063 — two-timescale latent (NULL, and DEGENERATE) + per-team sigma (WORSE)

Stopped at window 255/343 (74%) by choice: the remaining tail would have cost ~5 more hours
for the last fifth of the data, and the close-line metric resolves direction at ~5x power.
Both arms cover the identical 255 windows, so the comparison is paired. 510 fits, 0 failed,
0 corrupt checkpoints.

| arm | RPS | dRPS vs OUTCOME | dRPS vs CLOSE | LOSO |
|---|---|---|---|---|
| champion | 0.19634 | — | — | — |
| `FORM` (two-timescale) | 0.19638 | +0.00003 [−0.00012,+0.00019] | −0.000017 [−0.000046,+0.000012] | 3/8 |
| `PTSIG` (per-team sigma) | 0.19673 | +0.00039 [−0.00002,+0.00081] | **+0.000172 [+0.000109,+0.000237] WORSE** | 3/8 |

## THE TWO TIMESCALES DID NOT SEPARATE — the decomposition is DEGENERATE

This is the finding, and the reason the diagnostic was added before launching:

| parameter | posterior mean | prior intent |
|---|---|---|
| `rho_att_ability` | 0.9860 | ~0.98 (persistent) |
| `rho_att_form` | **0.9754** | ~0.89 (mean-reverting) |
| `rho_def_ability` | 0.9719 | ~0.98 |
| `rho_def_form` | **0.9727** | ~0.89 |
| `sigma_att_ability` | 0.0029 | 0.003 |
| `sigma_att_form` | 0.0227 | 0.015 |

**The data pulled `form`'s persistence UP to match `ability`'s** — on the defence side they
are identical to three decimals, with form fractionally HIGHER. The sigmas separated (form
carries ~8x the innovation SD) but the timescales did not. Two AR(1) processes with the same
rho and different sigmas are mathematically ONE AR(1) with
`sigma = sqrt(0.0029^2 + 0.0227^2) = 0.0229`.

So `FORM` is not a two-timescale model that failed. It is a model that **could not BE
two-timescale**: the structure is not identified by this data, it collapsed to a single walk
with a larger innovation SD, and the only consequence was a different total variance.

That also explains the conditioning damage: **50/255 windows above rhat 1.1 (19.6%) against
the champion's 14/343 (4.1%)**, median 1.065 vs 1.038. Two unidentified processes competing
to explain the same variation is textbook non-identifiability.

## Per-team sigma is WORSE, not null

`PTSIG` is +0.000172 on the close with the CI excluding zero. Letting innovation volatility
vary by team actively hurts. This confirms WP006's original rejection, now against the
CURRENT champion (xG 0.75 + deep + stakes) on 255 windows rather than the far older model it
was first judged against. The WP049 lead "promoted-team innovation variance" is closed.

## What this means for the structural direction

I had called this "the strongest remaining lead", on the reasoning that WP049 showed Elo's
advantage was in-season TRACKING SPEED and that tracking speed is what the innovation process
governs. That reasoning was sound and the answer is still no — and the degeneracy tells us
why: **this data cannot identify two timescales**, so you cannot buy faster tracking for real
shifts while keeping it slow for noise. One walk is all the data supports.

**The one variant left** would be to FORCE separation rather than let it be learned — pin
`rho_ability = 0.995` and `rho_form = 0.85` as constants instead of sampling them. That tests
the structure rather than the data's willingness to find it. Expected value is low (the data
actively prefers collapse, and prior/structure changes are now 0-for-~22), but it is the only
honest way to say the two-timescale idea itself was tested rather than its identifiability.

## Caveat

Truncated at 255/343. The dropped windows are the ones with the MOST training history, and a
benefit that grows with history would be understated. For `FORM` there is no mechanism for
that (it is degenerate); for `PTSIG`, which is worse, more data would have to reverse a
CI-excluding result.
