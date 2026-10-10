# archive — the durable record

Built by `scripts/build_archive.py`; **fidelity is asserted, not assumed** —
`python scripts/build_archive.py verify` rebuilds exactly what
`wp007_structural_fixes/analysis.py::load_checkpoints` returns for each of the 34 work
products and compares it against the live pickles, including bit-identical checks on the
lambda / probability / RPS columns. All 34 match.

Rebuild or re-check at any time:

    python scripts/build_archive.py build     # pickles -> archive/
    python scripts/build_archive.py verify    # archive == pickles, or exit 1

## Why this exists

Pickle is a fine cache and a bad record: version-fragile, binary, undiffable. Some of these
cost real money to recreate — the checkpoints are 150+ hours of MCMC, and the Transfermarkt
scrapes are ~2,700 polite requests against a third party that can restructure or block at
any time, making them **unrecoverable** rather than merely expensive.

Parquet+zstd gets 6-10% of pickle size, which is what makes committing viable: **105.8MB ->
23.9MB**. The earlier plan assumed the big raw assets were too large for git; that was
measured and wrong.

## Contents

| file | rows | what |
|---|---|---|
| `predictions.parquet` | 236,212 | every per-match CV prediction, all 34 work products, 81 arms. Columns as `load_checkpoints()[0]` plus `wp` |
| `window_diagnostics.parquet` | 8,219 | per-window rhat / divergences / timings / posterior means. `param_means` flattened to `param_*`, as the loader does |
| `raw/player_minutes.parquet` | 207,130 | Understat per-player per-match minutes, position, importance |
| `raw/odds_1x2.parquet` | 12,704 | football-data 1X2 open + close, referee, half-time scores |
| `raw/odds_totals_ah.parquet` | 12,704 | O/U 2.5 and Asian handicap, open + close |
| `raw/odds_all_leagues.parquet` | 19,837 | five leagues incl. match stats; `__source` names the league |
| `raw/elo_ratings.parquet` | 273,972 | club Elo time series |

The scraped Transfermarkt JSON is **not duplicated here**. It is small, text and diffable, so
Parquet buys nothing and a second copy would be two sources that drift. It is tracked where
the scrapers write it: `work_products/wp059_injuries/{injuries,squads}.json` (16,808 spells,
2,989 players) and `work_products/wp065_market_value/values.json` (7,139 player-seasons).

**The rule:** bulky or fragile formats are converted here and the original is gitignored;
small text stays tracked at source. So `elo_ratings.csv` (11MB) is ignored in favour of
`raw/elo_ratings.parquet` (0.8MB), and the pickles are ignored in favour of the two
checkpoint Parquets.

## Loading

```python
import pandas as pd
P = pd.read_parquet("archive/predictions.parquet")
champ = P[(P.wp == "wp055_horizon1") & (P.arm == "H1_CHAMP")]
```

`config_overrides` and any other dict-valued column are stored as JSON strings — use
`json.loads` on them.

## What is deliberately NOT here

Derived intermediates that regenerate in minutes from the above plus the work-product code:
`cv_shared_data_*.pkl` (52MB) and `features*.pkl`. Keeping them would triple the archive for
no durability gain.
