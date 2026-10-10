"""WP065 — Transfermarkt squad MARKET VALUES, the one quality signal not derived from results.

WHY THIS AND NOT ANOTHER MODEL CHANGE. The latents absorb everything visible in past
results, which is why ~22 prior/structure arms and 228 engineered features all came back
null. Market value is a forward-looking assessment that does NOT come from this league's
match outcomes, so it is information the walk cannot already hold.

THE SPECIFIC GAP IT COULD FILL. WP059's working injury feature weights absent players by
their SHARE OF RECENT MINUTES. A GBP100m signing injured in his first month has almost no
minutes history, so he is weighted ~0 -- the feature is blind to exactly the absences that
should matter most. Market value is available from the moment he signs.

PARSING IS ROW-SCOPED, deliberately. A regex that matches a player id and then a value cell
with `.*?` across the page can silently pair a player with ANOTHER ROW's value -- the same
class of error that produced WP043's fake result. Here the page is split into `<tr>` rows
FIRST and the pairing happens only within a row. Validated before scraping: Liverpool 2019
totals EUR 1,003m, and Salah/Mane EUR 120m, Alexander-Arnold EUR 99m, van Dijk EUR 80m all
present with the right values and correct ordering.
"""
import json, os, re, sys, time
from pathlib import Path

B = os.path.abspath(".certs/bundle.pem")
os.environ["SSL_CERT_FILE"] = B
os.environ["REQUESTS_CA_BUNDLE"] = B
import requests                                # noqa: E402

HERE = Path(__file__).resolve().parent
WP = HERE.parent
SEASONS = list(range(2016, 2026))
PAUSE = 1.3
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120 Safari/537.36",
      "Accept-Language": "en-US,en;q=0.9"}
S = requests.Session(); S.headers.update(UA)
OUT = HERE / "values.json"


def eur_m(v):
    """'EUR72.00m' / 'EUR900k' / 'EUR1.10bn' -> millions of euros. '-' -> None."""
    v = v.replace("€", "").replace(",", "").strip()
    if not v or v == "-":
        return None
    try:
        if v.endswith("bn"):
            return float(v[:-2]) * 1000.0
        if v.endswith("m"):
            return float(v[:-1])
        if v.endswith("k"):
            return float(v[:-1]) / 1000.0
        return float(v)
    except ValueError:
        return None


def get(url, tries=3):
    for k in range(tries):
        try:
            r = S.get(url, timeout=30)
            if r.status_code == 200:
                return r.text
            if r.status_code in (404, 410):
                return None
        except Exception:                      # noqa: BLE001
            pass
        time.sleep(3 * (k + 1))
    return None


def main():
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for yr in SEASONS:
        key = str(yr)
        if key in out:
            print(f"  {yr}: cached ({len(out[key])} players)", flush=True); continue
        t = get("https://www.transfermarkt.com/premier-league/startseite/wettbewerb/"
                f"GB1/plus/?saison_id={yr}")
        clubs = sorted(set(re.findall(
            rf'/([a-z0-9\-]+)/startseite/verein/(\d+)/saison_id/{yr}', t or "")))
        assert len(clubs) == 20, f"{yr}: expected 20 clubs, got {len(clubs)}"
        rows = []
        for slug, cid in clubs:
            time.sleep(PAUSE)
            ct = get(f"https://www.transfermarkt.com/{slug}/startseite/verein/{cid}/saison_id/{yr}")
            if ct is None:
                print(f"    !! {yr} {slug}: fetch failed", flush=True); continue
            n0 = len(rows)
            for r in re.split(r'<tr class="(?:odd|even)">', ct)[1:]:
                pid = re.search(r'/([a-z0-9\-]+)/profil/spieler/(\d+)', r)
                val = re.search(r'<td class="rechts hauptlink">\s*<a[^>]*>\s*([^<]+?)\s*</a>', r)
                if not (pid and val):
                    continue
                v = eur_m(val.group(1))
                if v is None:
                    continue
                rows.append({"pid": pid.group(2), "slug": pid.group(1),
                             "club": slug, "value_m": v})
            if len(rows) == n0:
                print(f"    !! {yr} {slug}: 0 values parsed", flush=True)
        out[key] = rows
        OUT.write_text(json.dumps(out))
        tot = sum(r["value_m"] for r in rows)
        print(f"  {yr}: {len(rows)} players, league total EUR {tot:,.0f}m "
              f"(mean club EUR {tot/20:,.0f}m)", flush=True)
    n = sum(len(v) for v in out.values())
    print(f"\nPHASE done: {n} player-season values across {len(out)} seasons")


if __name__ == "__main__":
    main()
