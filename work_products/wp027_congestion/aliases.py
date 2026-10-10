"""openfootball club names -> the project's EPL codes.

openfootball uses full registered names; football-data (and so CODE_TO_FD) uses short
ones. 20 of our 35 clubs differ, and an unmatched club loses its cup/European fixtures
entirely -- which makes its rest days WRONG, not missing. So this map is explicit and
validated rather than fuzzy.
"""
OF_TO_CODE = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU", "Brentford": "BRE",
    "Brighton & Hove Albion": "BRI", "Burnley": "BUR", "Cardiff City": "CAR",
    "Chelsea": "CHE", "Middlesbrough": "MID", "Crystal Palace": "CRY", "Everton": "EVE", "Fulham": "FLH",
    "Huddersfield Town": "HUD", "Hull City": "HUL", "Ipswich Town": "IPS",
    "Leeds United": "LED", "Leicester City": "LEI", "Liverpool": "LIV",
    "Luton Town": "LUT", "Manchester City": "MCI", "Manchester United": "MUN",
    "Newcastle United": "NEW", "Norwich City": "NOR", "Nottingham Forest": "NOT",
    "Queens Park Rangers": "QPR", "Sheffield United": "SHE", "Southampton": "SOU",
    "Stoke City": "STO", "Sunderland": "SUN", "Swansea City": "SWA",
    "Tottenham Hotspur": "TOT", "Watford": "WAT", "West Bromwich Albion": "WBA",
    "West Ham United": "WHU", "Wolverhampton Wanderers": "WOL",
}


def to_code(name):
    """Tolerates the affixes openfootball attaches. AFC is a PREFIX for Bournemouth
    ("AFC Bournemouth") and a SUFFIX elsewhere ("Barrow AFC"), so both are stripped."""
    n = name.strip()
    for suf in (" FC", " AFC", " CF"):
        if n.endswith(suf):
            n = n[: -len(suf)].strip()
    for pre in ("AFC ", "FC "):
        if n.startswith(pre):
            n = n[len(pre):].strip()
    return OF_TO_CODE.get(n)
