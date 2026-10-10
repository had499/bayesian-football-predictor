"""Convert the project's pickles into a compact, durable, committable archive.

WHY. Pickle is a fine cache and a bad record: it is version-fragile (a pandas or numpy
upgrade can make a file unreadable), binary, and undiffable. Several of these artefacts cost
real money to recreate -- 8,219 MCMC checkpoints represent well over 150 hours of sampling,
and the Transfermarkt scrapes took ~2,700 polite HTTP requests against a third party that can
restructure or block at any time.

WHAT CHANGED vs the BACKLOG plan. The backlog said the big raw binaries were "too large for
git, back up externally". Measured rather than assumed, Parquet+zstd gets them to 6-10% of
pickle size, which moves them comfortably inside git:

    player_minutes      18.8MB -> 1.9MB      odds_all_leagues  25.9MB -> 2.2MB
    odds_raw             3.2MB -> 0.3MB      odds_other         4.2MB -> 0.2MB
    8,219 checkpoints   39.6MB -> 14.9MB     and 8,219 files   -> 2

FIDELITY IS ASSERTED, NOT ASSUMED. `verify` rebuilds exactly what
`wp007_structural_fixes.analysis.load_checkpoints` returns for each work product and compares
it column-by-column against the live pickles. The archive is only useful if it is provably
the same data, so a mismatch fails loudly.

The pickles stay on disk as the working cache (gitignored). The archive is the RECORD.
"""
import argparse, glob, json, pickle, sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
WP = REPO / "work_products"
ARCHIVE = REPO / "archive"
COMP = dict(compression="zstd", index=False)

# raw/scraped assets worth keeping: expensive or impossible to re-acquire
RAW = [
    ("wp030_squad_intactness/player_minutes.pkl", "player_minutes.parquet", "frame"),
    ("wp008_market_benchmark/odds_raw.pkl",       "odds_1x2.parquet",       "frame"),
    ("wp041_other_markets/odds_other.pkl",        "odds_totals_ah.parquet", "frame"),
    ("wp044_league_scan/odds_all_leagues.pkl",    "odds_all_leagues.parquet", "dict"),
    ("wp025_elo_channel/elo_ratings.csv",         "elo_ratings.parquet",    "csv"),
]
# text sources: already small and diffable, copied verbatim rather than converted
JSONS = [
    ("wp059_injuries/injuries.json", "injuries.json"),
    ("wp059_injuries/squads.json",   "squads.json"),
    ("wp065_market_value/values.json", "market_values.json"),
]


def _stringify(df):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].apply(lambda v: json.dumps(v, default=str)
                                if isinstance(v, (dict, list)) else v).astype("string")
    return df


def checkpoint_frames():
    """Exactly what load_checkpoints() produces, plus a `wp` column, for every work product.

    Mirrors analysis.load_checkpoints: param_means is FLATTENED to param_* columns (not
    dropped), everything else in `results` is kept as-is.
    """
    preds, wins = [], []
    for f in sorted(glob.glob(str(WP / "*" / "checkpoints" / "*" / "w*.pkl"))):
        p = Path(f)
        wp_name, arm = p.parts[-4], p.parts[-2]
        ck = pickle.loads(p.read_bytes())
        for r in ck.get("results", []):
            wins.append({"wp": wp_name, "arm": arm,
                         **{k: v for k, v in r.items() if k != "param_means"},
                         **{f"param_{k}": v for k, v in (r.get("param_means") or {}).items()}})
        for m in ck.get("cv_match_predictions", []):
            preds.append({"wp": wp_name, "arm": arm, **m})
    return pd.DataFrame(preds), pd.DataFrame(wins)


def build():
    ARCHIVE.mkdir(exist_ok=True)
    (ARCHIVE / "raw").mkdir(exist_ok=True)
    before = after = 0
    print("=== checkpoints ===")
    files = glob.glob(str(WP / "*" / "checkpoints" / "*" / "w*.pkl"))
    src = sum(Path(f).stat().st_size for f in files)
    P, W = checkpoint_frames()
    for df, name in ((P, "predictions.parquet"), (W, "window_diagnostics.parquet")):
        out = ARCHIVE / name
        _stringify(df).to_parquet(out, **COMP)
        print(f"  {name:28s} {df.shape}  {out.stat().st_size/1e6:6.2f}MB")
        after += out.stat().st_size
    before += src
    print(f"  ({len(files)} pickles, {src/1e6:.1f}MB -> 2 files)")

    print("\n=== raw / scraped assets ===")
    for rel, name, kind in RAW:
        p = WP / rel
        if not p.exists():
            print(f"  {rel}: MISSING, skipped"); continue
        if kind == "csv":
            df = pd.read_csv(p)
        else:
            obj = pd.read_pickle(p)
            if kind == "dict":
                df = pd.concat([v.assign(__source=k) for k, v in obj.items()],
                               ignore_index=True)
            else:
                df = obj
        out = ARCHIVE / "raw" / name
        _stringify(df).to_parquet(out, **COMP)
        before += p.stat().st_size; after += out.stat().st_size
        print(f"  {name:28s} {str(df.shape):16s} "
              f"{p.stat().st_size/1e6:6.1f}MB -> {out.stat().st_size/1e6:5.2f}MB")

    print("\n=== scraped JSON (copied verbatim: already text and diffable) ===")
    for rel, name in JSONS:
        p = WP / rel
        if not p.exists():
            print(f"  {rel}: MISSING, skipped"); continue
        (ARCHIVE / "raw" / name).write_bytes(p.read_bytes())
        before += p.stat().st_size; after += p.stat().st_size
        print(f"  {name:28s} {p.stat().st_size/1e6:5.2f}MB")

    print(f"\nTOTAL {before/1e6:.1f}MB  ->  {after/1e6:.1f}MB  ({after/before:.0%})")


def verify():
    """Assert the archive reproduces load_checkpoints() exactly, per work product."""
    sys.path.insert(0, str(WP / "wp007_structural_fixes"))
    from analysis import load_checkpoints
    P = pd.read_parquet(ARCHIVE / "predictions.parquet")
    W = pd.read_parquet(ARCHIVE / "window_diagnostics.parquet")
    wps = sorted({Path(f).parts[-4] for f in
                  glob.glob(str(WP / "*" / "checkpoints" / "*" / "w*.pkl"))})
    bad = 0
    print(f"verifying {len(wps)} work products against the live pickles\n")
    for name in wps:
        m, w = load_checkpoints(WP / name / "checkpoints")
        am = P[P.wp == name].drop(columns=["wp"]).reset_index(drop=True)
        key = ["arm", "window", "round", "home", "away"]
        num = [c for c in ("lambda_home", "lambda_away", "p_home", "p_draw", "p_away", "rps")
               if c in m.columns]
        a = m.sort_values(key).reset_index(drop=True)
        b = am.sort_values(key).reset_index(drop=True)
        ok = len(a) == len(b)
        if ok and num:
            ok = bool((a[num].to_numpy() == b[num].to_numpy()).all())
        aw = W[W.wp == name]
        okw = len(aw) == len(w)
        flag = "ok" if (ok and okw) else "MISMATCH"
        if not (ok and okw):
            bad += 1
        print(f"  {name:34s} preds {len(a):6d}/{len(b):6d}  windows "
              f"{len(w):5d}/{len(aw):5d}  {flag}")
    print(f"\n{'ALL MATCH — archive is faithful' if bad == 0 else f'{bad} WORK PRODUCTS MISMATCH'}")
    return bad == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["build", "verify", "both"], default="both", nargs="?")
    a = ap.parse_args()
    if a.action in ("build", "both"):
        build()
    if a.action in ("verify", "both"):
        print()
        sys.exit(0 if verify() else 1)
