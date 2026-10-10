"""Fetch per-match shot data from Understat and cache it.

WP014 needs xG split by situation — specifically non-penalty xG (npxG). The
roster data WP011 cached carries no penalty flag, so this needs the shot
endpoint. Same provider, deeper endpoint again; no new vendor.

Behind TLS interception, first:
    security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain > ca.pem
    security find-certificate -a -p /Library/Keychains/System.keychain >> ca.pem
    export REQUESTS_CA_BUNDLE=$PWD/ca.pem SSL_CERT_FILE=$PWD/ca.pem

Resumable: one cache file per season, completed matches skipped.

    python work_products/wp014_npxg_decomposition/fetch_shots.py
"""
import argparse
import pickle
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "shots"
SEASONS = [str(y) for y in range(2014, 2026)]   # full history — the model trains from 2014
SLEEP = 0.4


def _shot(x):
    """Keep only what downstream work needs. `minute` was omitted from the first
    version of this cache, which silently broke any score-state analysis — every
    goal looked like minute 0. X/Y are kept now as well so shot-location work
    never needs a third full re-fetch."""
    return {"xG": float(x["xG"]), "situation": x["situation"], "result": x["result"],
            "minute": int(float(x.get("minute") or 0)),
            "X": float(x.get("X") or 0.0), "Y": float(x.get("Y") or 0.0)}


def _complete(rec):
    """True if a cached record already carries the current field set, so a re-run
    tops up an older thinner cache instead of silently keeping it."""
    for side in ("h", "a"):
        for sh in rec.get(side, []):
            return "minute" in sh
    return True          # a match with no shots at all is fine as-is


def fetch_season(client, season, sleep=SLEEP):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{season}.pkl"
    out = pickle.loads(path.read_bytes()) if path.exists() else {}
    matches = client.league(league="EPL").get_match_data(season)
    todo = [m for m in matches
            if m.get("isResult") and (m["id"] not in out or not _complete(out[m["id"]]))]
    print(f"[{season}] {len(matches)} matches, {len(out)} cached, {len(todo)} to fetch", flush=True)

    for i, m in enumerate(todo, 1):
        mid = m["id"]
        try:
            shots = client.match(match=str(mid)).get_shot_data()
        except Exception as e:  # noqa: BLE001
            print(f"  [{season}] {mid} FAILED {type(e).__name__} {str(e)[:70]}", flush=True)
            continue
        # keep only the fields downstream work needs (see _shot)
        out[mid] = {
            "meta": {"match_id": mid, "datetime": m["datetime"], "season": season,
                     "home_short": m["h"]["short_title"], "away_short": m["a"]["short_title"]},
            "h": [_shot(x) for x in shots.get("h", [])],
            "a": [_shot(x) for x in shots.get("a", [])],
        }
        if i % 25 == 0 or i == len(todo):
            path.write_bytes(pickle.dumps(out))
            print(f"  [{season}] {i}/{len(todo)}", flush=True)
        time.sleep(sleep)

    path.write_bytes(pickle.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", default=SEASONS)
    ap.add_argument("--sleep", type=float, default=SLEEP)
    args = ap.parse_args()
    from understatapi import UnderstatClient
    c = UnderstatClient()
    total = sum(len(fetch_season(c, s, args.sleep)) for s in args.seasons)
    print(f"done: {total} matches cached -> {CACHE}")


if __name__ == "__main__":
    main()
