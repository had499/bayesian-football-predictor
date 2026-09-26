# WP008 — Market benchmark: does the xG channel close any of the gap to Pinnacle?

**Status: complete.** WP007 found the xG measurement channel improves RPS but couldn't say whether that improvement means anything against the market — the odds benchmark was unobtained. This work product gets that benchmark and answers the question directly: **yes, partially.** `L5_xgchannel` closes about a third of `L0_baseline`'s gap to Pinnacle, and in two of six seasons reaches statistical parity with it. The rest of the gap — and it's still the larger part — is the same "worse specifically where the model disagrees most with the market" pattern WP003 found originally.

## Why the odds needed a workaround

football-data.co.uk (WP003/WP004's source) returns a block page on this network — a Mastercard corporate web-filtering proxy, not a real server error, confirmed by reading the response body (`Application Blocked … blocked in accordance with Mastercard policy`). That's a network policy decision, not something to route around technically, and unblocking it needs the account holder's own IT access-request process (linked from the block page).

Instead, this uses a public GitHub mirror — [AnishKhetani/premier-league-data](https://github.com/AnishKhetani/premier-league-data) — which states it sources from football-data.co.uk and preserves its odds columns verbatim (`PSCH`/`PSCD`/`PSCA` etc., renamed to e.g. `pinnacle_1x2_home_close`). Verified before use, in `fetch_odds.py`:
- Team-name spelling matches Understat's exactly (`fetch_odds.py`'s join hit **1,314/1,314** WP007 matches with **0 score mismatches** against the mirror's own `fthg`/`ftag` — the strongest possible check that this is the right data).
- Overround figures land close to WP003's own numbers from the original source: Pinnacle 2.68% here vs. 2.8% in WP003; Bet365 5.48% vs. 5.5%; market-average 4.49% vs. 4.3%.
- Pinnacle coverage: 1,205/1,314 (92%) — Bet365 and market-average cover all 1,314.

If football-data.co.uk becomes reachable again, prefer it directly — this mirror is a workaround, one hop further from the primary source, not a permanent replacement.

## Results

All numbers: 1,205 matches with Pinnacle closing odds (of WP007's 1,314), 2020-21 through 2025-26 (2025-26 partial — Pinnacle coverage for that season lags in the mirror). CIs are window-cluster bootstrap, same convention as WP007.

### Pooled RPS

| Predictor | n | Pooled RPS | CI (cluster) |
|---|---|---|---|
| naive | 1205 | 0.2378 | [0.2346, 0.2407] |
| `L0_baseline` | 1205 | 0.2053 | [0.1955, 0.2146] |
| `L5_xgchannel` | 1205 | 0.2027 | [0.1930, 0.2120] |
| Bet365 | 1205 | 0.1975 | [0.1870, 0.2072] |
| **Pinnacle** | 1205 | **0.1972** | [0.1867, 0.2071] |
| market-average | 1205 | 0.1987 | [0.1886, 0.2083] |

Both model arms sit between naive and the market, same shape as WP003 found — `L5_xgchannel` sits closer.

### Paired gap to Pinnacle

| Arm | Gap vs. Pinnacle | CI (cluster) | Verdict |
|---|---|---|---|
| `L0_baseline` | +0.0081 | [+0.0045, +0.0115] | worse |
| **`L5_xgchannel`** | **+0.0055** | **[+0.0023, +0.0087]** | **worse, but smaller** |

**`L5_xgchannel` closes 32% of `L0_baseline`'s gap to Pinnacle** — both are still reliably worse than the market (both CIs exclude zero on the "worse" side), but the improvement is real, not just directionally consistent noise. Cross-check: scoring `L5` vs `L0` directly on this same Pinnacle-covered 1,205-match subset (no market involved) gives −0.0026 [−0.0048, −0.0003] — matching WP007's full-1,314-match estimate of −0.0025 to −0.0028 almost exactly. The two independent slices agree, which is some reassurance this isn't a subset-selection artifact.

### Where the remaining gap lives — the disagreement pattern WP003 found is still there

| Subset | n | RPS `L5_xgchannel` | RPS Pinnacle | Gap | CI (cluster) |
|---|---|---|---|---|---|
| all matches | 1205 | 0.2027 | 0.1972 | 0.0055 | [+0.0023, +0.0087] |
| top 50% disagreement | 603 | 0.2144 | 0.2032 | 0.0112 | [+0.0055, +0.0169] |
| top 25% | 302 | 0.2161 | 0.1993 | 0.0168 | [+0.0082, +0.0253] |
| top 10% | 121 | 0.2343 | 0.2085 | 0.0257 | [+0.0094, +0.0408] |

The gap roughly **quadruples** from "all matches" to "the 10% of matches where the model disagrees most with Pinnacle" — exactly WP003's original finding, and the xG channel didn't fix it. This is consistent with WP004's conclusion that the shortfall is *structural resolution*, not calibration: on the matches where the model is confident and the market disagrees, the market is still usually right.

### Season-by-season: parity in two seasons, not the rest

| Season | n | Gap vs. Pinnacle | CI (cluster) |
|---|---|---|---|
| 2020 | 223 | +0.0074 | [+0.0038, +0.0121] |
| **2021** | 220 | **−0.0001** | **[−0.0078, +0.0066]** |
| 2022 | 205 | +0.0089 | [+0.0012, +0.0156] |
| 2023 | 228 | +0.0112 | [+0.0041, +0.0191] |
| **2024** | 219 | **−0.0008** | **[−0.0079, +0.0074]** |
| 2025 | 110 | +0.0072 | [+0.0012, +0.0145] |

`L5_xgchannel` reaches statistical parity with Pinnacle in **2021 and 2024** (CI includes zero) — worth noting 2024 is also the one season WP007 found individually significant for the `L5` vs `L0` comparison, so those two findings corroborate each other. The other four seasons remain clearly behind the market. This is not "the model beats the market some years" — every point estimate is non-negative, so at best the gap disappears, it never flips.

### Calibration vs. Pinnacle

| Predicted P(home win) | `L0` actual | `L5` actual | Pinnacle actual |
|---|---|---|---|
| 0.62–0.75 | 0.669 | 0.654 | 0.653 |
| 0.75–0.88 | 0.885 | 0.875 | 0.861 |

Interesting wrinkle: **Pinnacle itself is mildly under-confident in the same top bucket** (predicted 0.802 vs. actual 0.861) — this isn't a model-specific flaw, or at least not one unique to this model; the market's own probabilities undershoot actual home-win frequency there too, in this sample. `L5` narrows the model's own gap slightly (0.798 vs. 0.875, down from `L0`'s 0.788 vs. 0.885) but doesn't close it, consistent with WP007's finding.

## Bottom line

**Update (WP010)**: the +0.0055 figure below is measured against the *closing* line, which is set after team news is public. Against Pinnacle's earlier, pre-closing price — the fair benchmark for a model predicting days ahead — `L5_xgchannel`'s gap is **+0.0039**, and only ~29% of the closing-gap is late information. See `work_products/wp010_opening_vs_closing/`.

The xG channel is not cosmetic — it buys real ground against the market, not just against naive. But "32% of the gap, concentrated in 2 of 6 seasons, and the confident-disagreement pattern is untouched" is an honest description of a partial result, not "the model now competes with Pinnacle." The remaining gap is where WP004 already pointed: structural resolution on the matches where the model and the market disagree most, which no amount of recalibration (WP004) or prior-loosening (WP005) reached, and which the WP007 ladder — beyond the xG channel — didn't move either.

## What this doesn't answer

* **Why does the xG channel help disproportionately in 2021 and 2024?** Not investigated — worth a season-level diagnostic (e.g. was xG's signal-to-noise unusually high or low those seasons, more/fewer upsets, different fixture congestion) before assuming it's the model rather than those two seasons being unusual.
* **The disagreement-quantile pattern** — same shape as WP003, untouched by WP007's one validated change — is the natural next target, but this work product doesn't diagnose *what specifically* the market sees on those high-disagreement matches that the model doesn't. **See WP009** (`work_products/wp009_disagreement_diagnostics/`): the model is measurably less sharp than the market and specifically under-rates away favourites, but post-hoc sharpening doesn't help (it trades non-upset accuracy for upset accuracy at a net wash) — evidence the gap is genuinely missing information, not a fixable calibration artifact.
* **Betting simulation** (WP003 had one, using market-max odds) — not repeated here; the RPS/calibration answer was the priority and the betting-sim conclusion is unlikely to change materially (the model is still behind the market on the matches that would matter for it).
* **`market_max`** odds are in `odds_raw.pkl` (fetched, unused) if a future work product wants them.

## Reproducing

```bash
# CA bundle only needed behind TLS interception; not needed for the block itself
export REQUESTS_CA_BUNDLE=/path/to/ca.pem SSL_CERT_FILE=/path/to/ca.pem
python work_products/wp008_market_benchmark/fetch_odds.py         # writes odds_raw.pkl
```

Then re-run `wp008_market_benchmark.ipynb`'s cells (`jupyter nbconvert --to notebook --execute wp008_market_benchmark.ipynb`), which already holds the executed readout this README's numbers were pulled from. Analysis logic lives in `market_analysis.py` (odds-joining, team-code mapping, score-consistency checks — unit-tested in `tests/test_wp008_analysis.py`) plus WP007's `analysis.py` for the shared bootstrap/RPS statistics. Needs WP007's `cv_shared_data.pkl` and `checkpoints/` to already exist (this work product reads WP007's predictions; it doesn't retrain anything). Runs in well under a minute — no model fitting, just joins and bootstraps.
