# WP057 — the CLOSING LINE is a ~5x higher-power yardstick than the realised outcome

Prompted by the user's question: "we're modelling the open price not the CLV — does this
mean we're optimising for the wrong thing?"

**The objective is NOT wrong.** RPS against realised outcomes is a proper scoring rule and
the right target. The open-price constraint governs what the model may CONSUME, which is a
separate matter. But the **measuring instrument** has been needlessly noisy, and that is
very likely why ~25 arms came back "null".

## The argument

One match is one coin flip, so RPS-vs-outcome carries irreducible noise. In a PAIRED
comparison it cancels only to the extent the two arms agree, which for small
modelling changes is not much. MDE is +-0.00095 at 115 windows.

The closing line is a low-variance estimate of the same truth. Define the SAME ordinal RPS
with the close's cumulative probabilities as a soft target in place of the 0/1 outcome:

    RPS_close(model) = mean_i sum_{k<3} ( cumsum(p_model)_k - cumsum(p_close)_k )^2 / 2

If p_close ~= truth then `E[RPS_outcome] = RPS_close + irreducible(p_close)`, and the
second term does not depend on the model — so comparing arms on RPS_close estimates the
same contrast with the outcome noise removed. Close is Shin-de-vigged Pinnacle.

## Validation (n=3,597; `close_as_yardstick.py`)

| pair | dRPS vs OUTCOME | dRPS vs CLOSE | power | dir |
|---|---|---|---|---|
| deep ON vs OFF *(established, 0/10 LOSO)* | −0.00052 [−0.00169,+0.00064] | **−0.00047 [−0.00069,−0.00026]** | 5.5x | OK |
| xG weight .75 vs .25 *(established)* | −0.00085 [−0.00151,−0.00019] | −0.00027 [−0.00041,−0.00013] | 4.8x | OK |
| xG weight .75 vs 1.00 | −0.00011 [−0.00046,+0.00024] | **−0.00023 [−0.00032,−0.00014]** | 4.0x | OK |
| deep weight .25 vs 1.00 | −0.00056 [−0.00145,+0.00033] | **−0.00065 [−0.00082,−0.00047]** | 5.1x | OK |
| market channel 1.0 vs .25 | −0.00039 [−0.00108,+0.00029] | −0.00030 [−0.00044,−0.00015] | 4.8x | OK |
| intactness variants *(both null)* | −0.00003 | +0.00001 | 4.6x | both 0 |

**4.0-5.5x tighter CIs, direction agreeing on every pair carrying signal.** The three bold
rows were UNRESOLVABLE on outcome-RPS and are decisive against the close. The deep channel
is the headline: it was only ever established by counting LOSO fold signs because its
outcome CI straddled zero.

## Three caveats, all load-bearing

1. **A compass, not a ruler.** Magnitudes do not transfer (xG weight: −0.00085 outcome vs
   −0.00027 close). Use it for DIRECTION and SIGNIFICANCE. Headline numbers stay on
   outcome-RPS.
2. **CIRCULAR for any market-consuming arm.** The market channel is graded against the very
   thing it ingests. Its close-gain is proportionally LARGER than its outcome-gain
   (ratio 0.77 vs xG's 0.32) — that is the inflation showing. **Never screen a
   market-fed arm on this metric.** Same objection as WP039's circularity.
3. **Its ceiling is the close.** It cannot show us beating the market and would PENALISE an
   arm that genuinely did. Valid only while we are clearly behind — we are +0.00282 behind
   the OPEN, so fine for now, but it stops being a valid compass near parity. The close also
   contains team news we structurally cannot see (WP010: ~29% of the gap), so RPS_close has
   a floor > 0; that offset is common to all arms and harmless for paired contrasts.

## Protocol going forward

**Screen on close, confirm survivors on outcome-RPS.** Excluding market-fed arms. Selecting
many arms on this metric alone would drift the model toward being a market copier, which is
exactly the failure mode caveat 3 describes.

## What it unlocks

- Re-screen the ~25 arms killed as "null" against +-0.00095 — several may be real at 5x power.
- **Immediate lead: deep weight.** 0.25 beats 1.00 by −0.00065 (CI excludes zero) while the
  champion sits at **0.5**. WP033 concluded "0.5 confirmed optimal (no change)" on
  outcome-RPS; that conclusion was reached at 1/5 the power and should be re-opened.

---

# WP057b — RE-SCREEN RESULTS (~40 within-sweep contrasts, `rescreen.py`)

Every past sweep re-scored against the close. Within-sweep contrasts only; market-fed
arms (wp039/wp051) excluded as circular. Cross-sweep contrasts against the champion
(`W075`) were used only where the original work product had already established that
comparison as valid (WP033's own `analyse.py` does exactly this).

## The one real finding: THE DEEP CHANNEL IS CONFIRMED

| contrast | vs OUTCOME | vs CLOSE |
|---|---|---|
| `H_deep_added` vs `S3_xgchannel_only` (wp021) | −0.00085 [−0.00184,**+0.00012**] | **−0.00083 [−0.00104,−0.00063]** |

Point estimates agree to within 0.00002; only the interval differs. **WP021's "deep channel
NOT confirmed at 115 windows" is overturned** — it was an underpowered null, not an absent
effect. This also settles CHAMPION.md's standing caveat that deep was "included on expected
value not proof". It is now proven.

## Shipped decisions, all CONFIRMED at 5x power (no change to the champion)

| constant | champion | verdict |
|---|---|---|
| xG channel weight | 0.75 | optimal — 0.25 and 1.00 both worse |
| deep channel weight | 0.50 | optimal — 0.00/0.75/1.00 worse, 0.25 tied (+0.00005, CI incl. 0) |
| season_start_window | 3 | confirmed — 8 is tied/worse |
| season_start_sigma_mult | 3.0 | confirmed — 1.0 and 2.0 both tied |
| init_scale | current | confirmed — 0.5 and 1.0 both clearly WORSE |
| HFA sd, sigma priors | current | confirmed — tight and loose both tied or worse |

**Correction to a lead stated earlier in this file:** the first pass flagged "deep weight
0.25 beats 1.00 while the champion sits at 0.5 — re-open it". Wrong. Tested against the
CHAMPION rather than against 1.00, 0.25 is merely TIED (+0.00005 [−0.00002,+0.00012]).
The error was reading a within-family contrast as if it were a contrast with the champion.
Same for "SSM_2 better than SSM_1" — both are tied with the champion's 3.0.

## Nulls upgraded to WELL-POWERED nulls (now definitively closed, not "underpowered")

congestion (rest, load, and both x effort) · squad intactness · split defence · crowd
regime / wide HFA prior · npxG, per-season kappa, shot-xG · time-varying HFA and league
level · `L3_randomwalk`, `L4_promoted`, `S1_xgweight1`, `S2_dc_only_fixed`

Previously "null at ±0.00095". Still null at ~5x that resolution. These are closed.

## Confirmed WORSE (previously filed as null)

`L1_posterior` (+0.00017) and `L2_level` (+0.00025) from the original WP007 ladder.
Also re-confirmed worse: time decay, outcome-level likelihood at weight >0.05, stakes
dispersion, deep OFF.

## The headline, stated honestly: ZERO new positive leads

5x power did **not** resurrect a single dead arm. It confirmed one real effect (deep),
validated every shipped constant, and definitively buried the rest. That is genuinely
valuable — it means the directions tried are **exhausted**, not merely under-measured, so
there is no point revisiting them with more windows.

MULTIPLICITY: ~40 contrasts at 95% implies ~2 expected false positives. Every marginal
call here points in the "change nothing" direction, so none of them affects a decision.
The deep result is far from marginal and its point estimate independently matches the
outcome metric.

**Live directions remaining: WP056's new feature families and the GBM wrapper.** Not the
latent/prior/structure space.

---

# WP057c — validating the instrument: is the close-metric gain market BIAS or real skill?

The stated risk in caveat 2 above was market *bias*, not market noise: an arm that
replicates a known market distortion scores well on close-distance while being worse on
truth. Draws are the obvious exposure. Two checks.

## 1. Is the close calibrated? Yes — but the draw row carries a warning

| outcome | close mean p | empirical rate | close − actual |
|---|---|---|---|
| HOME | 0.4420 | 0.4487 [0.4324,0.4651] | −0.0067 [−0.0221,+0.0091] |
| DRAW | 0.2382 | 0.2302 [0.2158,0.2442] | +0.0080 [−0.0063,+0.0221] |
| AWAY | 0.3198 | 0.3211 [0.3043,0.3380] | −0.0013 [−0.0169,+0.0144] |

Every CI includes zero, so no mis-pricing is *detectable*. But the test only excludes a
bias larger than ~2pp, and the DRAW point estimate is +0.80pp. Our champion sits at 0.2274,
i.e. **closer to the empirical rate than the close is**. So the metric would nudge draw mass
UP, away from truth. Direction of the leak confirmed; magnitude unknown from this test alone.

## 2. EXACT decomposition settles it

`RPS_close = ( ||mean d||² + mean||d − mean d||² ) / 2`, where `d_i = cumsum(p_i) −
cumsum(close_i)`. This splits exactly into **LEVEL** (systematic mean offset from the close
— the bias-leak channel, since an arm can score well merely by shifting its average toward
the market) and **DISCRIMINATION** (per-match deviation — genuine match-by-match agreement).

| arm | RPS_close | LEVEL | DISCRIM | mean draw p |
|---|---|---|---|---|
| `S3_xgchannel_only` | 0.004698 | 0.000004 | 0.004694 | 0.2342 |
| `H_deep_added` | 0.003866 | 0.000021 | 0.003845 | 0.2299 |
| **deep's change** | **−0.000832** | **+0.000017** | **−0.000849** | |

**102% of the deep channel's gain is DISCRIMINATION; the LEVEL term moves slightly
AGAINST it.** Deep shifted draw mass to 0.2299 — essentially the empirical rate (0.2302)
and AWAY from the close (0.2382) — was *penalised* on level for doing so, and still won
overall. That is the exact opposite of the bias leak.

## The general result: the bias-leak channel is numerically negligible

LEVEL is 0.000004–0.000021 against DISCRIMINATION of ~0.0047 — **three orders of
magnitude smaller**. A pure level shift toward the market can buy at most ~0.00002, far
below any decision threshold. The risk is real in principle and immaterial in practice,
because the metric is ~99.5% discrimination-driven.

**Standing guard for future use:** report the LEVEL/DISCRIM split alongside any
close-metric claim. If LEVEL carries a meaningful share of a gain, distrust that gain —
the arm is drifting toward the market's average rather than predicting matches.

---

# WP057d — the GBM WRAPPER IS REAL (`screen_gbm.py`)

The wrapper was the only live candidate left and was UNRESOLVED on outcome-RPS at
horizon 1: −0.00040 [−0.00083,**+0.00005**] at beta=0.5. Screened against the close
(legal — the GBM is not market-fed, unlike the market channel). n=1,388 matches with a
Pinnacle close, eval windows 201-343, config locked on 70-200.

| beta | dRPS vs OUTCOME | dRPS vs CLOSE | LEVEL | DISCRIM |
|---|---|---|---|---|
| 0.00 | (baseline) | (baseline) | 0.000043 | 0.003077 |
| 0.25 | −0.00019 [−0.00043,+0.00006] | **−0.00008 [−0.00011,−0.00004]** | 0.000044 | 0.003000 |
| **0.50** | −0.00033 [−0.00081,+0.00017] | **−0.00010 [−0.00016,−0.00003]** | 0.000045 | 0.002980 |
| 1.00 | −0.00044 [−0.00141,+0.00054] | +0.00004 [−0.00009,+0.00016] | 0.000049 | 0.003107 |

**CONFIRMED at beta=0.25 and 0.50** — CI excludes zero where outcome-RPS could not resolve
it. The WP057c guard passes cleanly: at beta=0.5 the gain decomposes as DISCRIMINATION
−0.000097 and LEVEL +0.000002, i.e. ~100% genuine per-match prediction, no drift toward the
market's average.

**beta=1.00 OVER-CORRECTS.** On outcome-RPS the point estimates rise monotonically with beta
(−0.00019 / −0.00033 / −0.00044), which reads as "more correction is better". The close
metric says the opposite above 0.5: +0.00004, CI including zero, and DISCRIM back ABOVE
baseline. At 5x power the close is the more trustworthy read, and beta=0.5 was already
WP053's pre-registered choice, so this is a confirmation rather than a fresh selection.
(Noting honestly that calling beta=1.0 "over-correcting" IS a post-hoc read of the eval
range; the beta=0.5 verdict is not.)

**This is the second previously-unresolvable question the close metric has settled**, after
the deep channel. Both had outcome point estimates agreeing in sign, which is the required
guard.
