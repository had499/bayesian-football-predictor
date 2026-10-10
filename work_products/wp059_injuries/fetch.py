"""WP059 — published, DATED injury spells from Transfermarkt.

WHY THIS AND NOT THE OTHER CANDIDATES. WP010 measured the remaining gap to the closing
line as late information. Of the sources probed 2026-10-09 (see
reference-data-source-landscape), this is the only reachable one that is (a) a PUBLISHED
absence signal rather than a proxy we derived ourselves, (b) DATED, so legality is decided
by arithmetic rather than assumption, and (c) carries SEVERITY (days out, games missed).
FBref 403s. The FPL archive's `chance_of_playing` is an END-OF-SEASON snapshot, useless
pre-match, and its gameweek transfer volumes are recorded at the deadline (~90 min pre-KO),
i.e. AFTER the opening line, so they violate the open-price rule.

LEGALITY. A spell is only usable for a match if it STARTED at least 2 days before kickoff,
so it was public before the opening line posted. Enforced in build.py, not here.

NAME MATCHING, deliberately constrained. WP043 was wrecked by fuzzy matching: three clubs
collapsed onto "Krasnodar", 25% of rows silently vanished, and the score validator reported
0 mismatches throughout. So: clubs are matched by a HAND-CHECKED 34-entry map, and players
are then matched by name ONLY WITHIN the same club-season, where a 45-man squad makes
collisions rare. Unmatched names are printed for inspection, never silently dropped.

Two phases, both cached to disk per page so the run is resumable:
  1. squads   20 clubs x 10 seasons = 200 pages -> player_id, name, club, season
  2. injuries one page per distinct player_id   -> dated spells
"""
import json, os, re, sys, time
from pathlib import Path

B = os.path.abspath(".certs/bundle.pem")
os.environ["SSL_CERT_FILE"] = B
os.environ["REQUESTS_CA_BUNDLE"] = B
import io                                      # noqa: E402
import pandas as pd                            # noqa: E402
import requests                                # noqa: E402

HERE = Path(__file__).resolve().parent
SEASONS = list(range(2016, 2026))
PAUSE = 1.3                                    # polite; ~2,700 pages ≈ 1 hour
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120 Safari/537.36",
      "Accept-Language": "en-US,en;q=0.9"}
S = requests.Session(); S.headers.update(UA)
SQUADS = HERE / "squads.json"
INJ = HERE / "injuries.json"


def get(url, tries=3):
    for k in range(tries):
        try:
            r = S.get(url, timeout=30)
            if r.status_code == 200:
                return r.text
            if r.status_code in (404, 410):
                return None
        except Exception:                       # noqa: BLE001
            pass
        time.sleep(3 * (k + 1))
    return None


def phase_squads():
    out = json.loads(SQUADS.read_text()) if SQUADS.exists() else {}
    for yr in SEASONS:
        key = str(yr)
        if key in out:
            print(f"  {yr}: cached ({len(out[key])} players)", flush=True); continue
        t = get("https://www.transfermarkt.com/premier-league/startseite/wettbewerb/"
                f"GB1/plus/?saison_id={yr}")
        clubs = sorted(set(re.findall(rf'/([a-z0-9\-]+)/startseite/verein/(\d+)/saison_id/{yr}', t or "")))
        assert len(clubs) == 20, f"{yr}: expected 20 clubs, got {len(clubs)}"
        rows = []
        # Extract from the HREF, not the anchor text. The `<td class="hauptlink">` pattern
        # SILENTLY DROPS real squad players -- on Liverpool 2019 it found 50 and missed
        # Virgil van Dijk, Alexander-Arnold, Alisson, Shaqiri and 7 more (href finds 61).
        # The href also carries the name as a slug, which is a cleaner matching key than
        # anchor text (no markup variation, and no special-character trouble: TM's
        # "Martin Odegaard" slug avoids the fact that U+00D8 does NOT NFKD-decompose, so
        # accent-stripping "Ødegaard" yields "degaard" and fails to match "Odegaard").
        # A superset is harmless here: we only ever look up players matched from Understat,
        # so extra non-squad links on the page cost nothing.
        for slug, cid in clubs:
            time.sleep(PAUSE)
            ct = get(f"https://www.transfermarkt.com/{slug}/startseite/verein/{cid}/saison_id/{yr}")
            if ct is None:
                print(f"    !! {yr} {slug}: fetch failed", flush=True); continue
            found = {(pid, pslug) for pslug, pid
                     in re.findall(r'/([a-z0-9\-]+)/profil/spieler/(\d+)', ct)}
            if not found:
                print(f"    !! {yr} {slug}: 0 players parsed", flush=True); continue
            for pid, pslug in found:
                rows.append({"pid": pid, "slug": pslug, "club": slug, "cid": cid})
        out[key] = rows
        SQUADS.write_text(json.dumps(out))
        print(f"  {yr}: {len(clubs)} clubs, {len(rows)} squad rows, "
              f"{len({r['pid'] for r in rows})} distinct players", flush=True)
    allp = {r["pid"] for v in out.values() for r in v}
    print(f"\nPHASE 1 done: {len(allp)} distinct players across {len(out)} seasons")
    return out


def phase_injuries(squads):
    pids = sorted({r["pid"] for v in squads.values() for r in v}, key=int)
    out = json.loads(INJ.read_text()) if INJ.exists() else {}
    todo = [p for p in pids if p not in out]
    failed = []
    print(f"PHASE 2: {len(out)} cached, {len(todo)} to fetch "
          f"(~{len(todo)*PAUSE/60:.0f} min)", flush=True)
    t0 = time.time()
    for i, pid in enumerate(todo, 1):
        time.sleep(PAUSE)
        t = get(f"https://www.transfermarkt.com/x/verletzungen/spieler/{pid}")
        spells = []
        if t:
            try:
                for tb in pd.read_html(io.StringIO(t)):
                    cols = [str(c) for c in tb.columns]
                    if "from" in cols and "until" in cols:
                        for _, r in tb.iterrows():
                            spells.append({"season": str(r.get("Season")),
                                           "injury": str(r.get("Injury")),
                                           "from": str(r.get("from")),
                                           "until": str(r.get("until")),
                                           "games": str(r.get("Games missed"))})
                        break
            except Exception:                   # noqa: BLE001
                # no tables at all (player never injured), or markup lxml cannot parse.
                # Either way an empty spell list is the correct, non-fatal answer -- but
                # record it so a systematic parse failure cannot masquerade as "never hurt".
                failed.append(pid)
        out[pid] = spells
        if i % 50 == 0 or i == len(todo):
            INJ.write_text(json.dumps(out))
            el = (time.time() - t0) / 60
            print(f"  [{i}/{len(todo)}] {sum(len(v) for v in out.values())} spells so far; "
                  f"{el:.0f}m elapsed, eta {el/i*(len(todo)-i):.0f}m", flush=True)
    INJ.write_text(json.dumps(out))
    n = sum(len(v) for v in out.values())
    print(f"\nPHASE 2 done: {n} spells for {len(out)} players "
          f"({sum(1 for v in out.values() if v)} have >=1)")
    print(f"  pages that FAILED TO PARSE: {len(failed)} "
          f"(these look like 'never injured' downstream -- check if large)")
    if failed:
        (HERE / "parse_failures.json").write_text(json.dumps(failed))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    sq = phase_squads()
    if which != "squads":
        phase_injuries(sq)
