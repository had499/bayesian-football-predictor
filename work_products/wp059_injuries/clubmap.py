"""Hand-checked Transfermarkt club slug -> Understat 3-letter code.

34 clubs on each side across 2016-2025, so a bijection must exist; both directions are
asserted below. This is written out by hand and never fuzzy-matched: WP043 was wrecked by
fuzzy club matching (three different clubs collapsed onto "Krasnodar", 25% of rows silently
dropped, and the score validator reported 0 mismatches the whole time).

The map is then VALIDATED independently by squad overlap in `validate()` — if a pair were
wrong, that club-season's Transfermarkt squad would not overlap the Understat player list.
"""
import re
import unicodedata

TM_TO_US = {
    "afc-bournemouth": "BOU",          "afc-sunderland": "SUN",
    "aston-villa": "AVL",              "brighton-amp-hove-albion": "BRI",
    "cardiff-city": "CAR",             "crystal-palace": "CRY",
    "fc-arsenal": "ARS",               "fc-brentford": "BRE",
    "fc-burnley": "BUR",               "fc-chelsea": "CHE",
    "fc-everton": "EVE",               "fc-fulham": "FLH",
    "fc-liverpool": "LIV",             "fc-middlesbrough": "MID",
    "fc-southampton": "SOU",           "fc-watford": "WAT",
    "huddersfield-town": "HUD",        "hull-city": "HUL",
    "ipswich-town": "IPS",             "leeds-united": "LED",
    "leicester-city": "LEI",           "luton-town": "LUT",
    "manchester-city": "MCI",          "manchester-united": "MUN",
    "newcastle-united": "NEW",         "norwich-city": "NOR",
    "nottingham-forest": "NOT",        "sheffield-united": "SHE",
    "stoke-city": "STO",               "swansea-city": "SWA",
    "tottenham-hotspur": "TOT",        "west-bromwich-albion": "WBA",
    "west-ham-united": "WHU",          "wolverhampton-wanderers": "WOL",
}
assert len(TM_TO_US) == 34, len(TM_TO_US)
assert len(set(TM_TO_US.values())) == 34, "not injective -- two TM clubs share a code"


# Characters that do NOT decompose under Unicode NFKD, so accent-stripping alone
# mangles them: "Hojbjerg" stays matched only if we map o/ explicitly, and "Gross"
# needs ss -> ss. Without this, "Pierre-Emile Hojbjerg" normalised to "hjbjerg" and
# never matched Transfermarkt's "pierre-emile-hojbjerg".
TRANSLIT = str.maketrans({
    "\u00f8": "o", "\u00d8": "O", "\u00df": "ss", "\u0111": "d", "\u0110": "D",
    "\u0131": "i", "\u0142": "l", "\u0141": "L", "\u00e6": "ae", "\u00c6": "AE",
    "\u0153": "oe", "\u0152": "OE", "\u00f0": "d", "\u00fe": "th", "\u0127": "h",
})


def norm(name: str) -> str:
    """accent- and punctuation-insensitive key for matching players WITHIN a club-season."""
    s = str(name).translate(TRANSLIT)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", s.lower())


def tokens(name: str) -> frozenset:
    """normalised name TOKENS, for the subset fallback."""
    s = str(name).translate(TRANSLIT)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return frozenset(t for t in re.split(r"[^a-z]+", s.lower()) if len(t) > 1)


# Hand-verified overrides for nickname/transliteration variants the three automatic passes
# cannot reach. Each was checked to be the SAME PERSON, not merely a close string:
#   Max Kilman = Maximilian Kilman; Matthew Cash = "Matty" Cash; Mat Ryan = Mathew Ryan;
#   Jose Holebas is Greek-registered as Jose Cholevas; Joseph/Joe Gomez; Nathaniel/Nat
#   Phillips; Joshua/Josh King; Vinicius/Vini Souza; Yeremi/Yeremy Pino; Mykhailo/Mykhaylo
#   Mudryk; Abduqodir/Abdukodir Khusanov; Alejandro "Alex" Jimenez; and Rayan Cherki's full
#   name is Mathis Rayan Cherki.
# DELIBERATELY NOT MAPPED, because the candidate is probably a DIFFERENT PLAYER:
#   "Louis Beyer" -> "jordan-beyer"      (Jordan Beyer is a separate footballer)
#   "Alvaro Fernandez" -> "alvaro-ferllo" (garbled slug, not confidently the same man)
#   "Franck Zambo" -> no candidate at all in his own squad page
# Leaving those three unmatched costs ~6,400 minutes and is the correct trade: WP043 shows
# what a confident-but-wrong identity match does to a result.
NAME_OVERRIDE = {
    "Max Kilman": "maximilian-kilman",      "Matthew Cash": "matty-cash",
    "Mat Ryan": "mathew-ryan",              "Jose Holebas": "jose-cholevas",
    "Joseph Gomez": "joe-gomez",            "Vinicius Souza": "vini-souza",
    "Alejandro Jimenez": "alex-jimenez",    "Yeremi Pino": "yeremy-pino",
    "Mykhailo Mudryk": "mykhaylo-mudryk",   "Nathaniel Phillips": "nat-phillips",
    "Abduqodir Khusanov": "abdukodir-khusanov", "Joshua King": "josh-king",
    "Mathis Cherki": "rayan-cherki",
}

def match_players(our_names, their_slugs):
    """our_names -> their slug, matched ONLY within one club-season.

    Three passes, each requiring a UNIQUE candidate so an ambiguous name is left
    unmatched rather than guessed (WP043: fuzzy matching silently collapsed three clubs
    into one and dropped 25% of rows while reporting no mismatches):
      1. exact normalised string
      2. token SUBSET either way -- handles "Ezri Konsa Ngoyo" vs "ezri-konsa",
         "Iyenoma Destiny Udogie" vs "destiny-udogie", and reversed order like
         "Son Heung-Min" vs "heung-min-son" (token sets ignore order)
      3. surname + first initial -- handles "Mat Ryan" vs "mathew-ryan"
    """
    out = {}
    avail = set(their_slugs)
    for nm in our_names:                      # hand-verified overrides take precedence
        key = norm(nm)
        for raw, slug in NAME_OVERRIDE.items():
            if norm(raw) == key and slug in avail:
                out[nm] = slug
    our_names = [n for n in our_names if n not in out]
    theirs_n = {}
    for sl in their_slugs:
        theirs_n.setdefault(norm(sl), []).append(sl)
    remaining = []
    for nm in our_names:
        c = theirs_n.get(norm(nm))
        if c and len(c) == 1:
            out[nm] = c[0]
        else:
            remaining.append(nm)
    tk = {sl: tokens(sl) for sl in their_slugs if sl not in out.values()}
    still = []
    for nm in remaining:
        t = tokens(nm)
        c = [sl for sl, ts in tk.items() if ts and (ts <= t or t <= ts)]
        if len(c) == 1:
            out[nm] = c[0]
        else:
            still.append(nm)
    for nm in still:
        t = tokens(nm)
        if not t:
            continue
        c = [sl for sl, ts in tk.items()
             if ts and (max(ts, key=len) == max(t, key=len))
             and {x[0] for x in ts} & {x[0] for x in t}]
        if len(c) == 1:
            out[nm] = c[0]
    return out
