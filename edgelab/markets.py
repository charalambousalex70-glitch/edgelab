"""Bookmaker prices, sharp fair probabilities and value bets.

Sources
  * football-data.co.uk: Bet365, Betfair Exchange, market average and best price,
    plus several named UK bookmakers. Collected Friday (weekend) and Tuesday (midweek).
  * The Odds API (optional, needs a free key in the ODDS_API_KEY secret): Pinnacle,
    Betfair Exchange and Betfair Sportsbook live, plus every UK bookmaker it lists.
  * Opta supercomputer: typed into opta.csv each week from Opta Analyst's preview.

Value rule ("Rule V", backtested on 2024/25 to 2026/27):
  sharp fair probability = no-margin Betfair Exchange price (averaged with no-margin
  Pinnacle when available). An outcome is value when the best bookmaker price beats
  the sharp fair odds by at least the threshold (2%): p_sharp x price - 1 >= 0.02.
  Exchanges and Pinnacle are the yardstick, so they are never the price we bet at.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUTCOMES = ("home", "draw", "away")
FD_SUFFIX = ("H", "D", "A")

# Football-Data column prefix -> display name. "Max" is the best price across the
# bookmakers Football-Data tracks; it does not say which bookmaker offered it.
FD_BOOKS = {"B365": "Bet365", "BFD": "Betfred", "BV": "BetVictor", "BW": "Bet&Win",
            "PP": "Paddy Power", "SKB": "Sky Bet", "WH": "William Hill", "LB": "Ladbrokes",
            "Max": "Best market price"}
FD_SHARP = {"BFE": "Betfair Exchange", "PS": "Pinnacle"}
FD_OTHER = {"Avg": "Market average"}

ODDS_API_SHARP = {"pinnacle": "Pinnacle", "betfair_ex_uk": "Betfair Exchange", "betfair_ex_eu": "Betfair Exchange"}
ODDS_API_EXCLUDE = {"matchbook", "smarkets"}  # other exchanges: not bookmaker prices

# The Odds API full names -> football-data names
TEAM_NAMES = {
    "AFC Bournemouth": "Bournemouth", "Brighton and Hove Albion": "Brighton", "Coventry City": "Coventry",
    "Hull City": "Hull", "Ipswich Town": "Ipswich", "Leeds United": "Leeds", "Leicester City": "Leicester",
    "Luton Town": "Luton", "Manchester City": "Man City", "Manchester United": "Man United",
    "Newcastle United": "Newcastle", "Nottingham Forest": "Nott'm Forest", "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves", "Sheffield United": "Sheffield United",
}


def no_vig(odds) -> list | None:
    """Fair probabilities from decimal odds, margin removed proportionally."""
    o = np.asarray(odds, float)
    if o.shape != (3,) or not np.isfinite(o).all() or (o <= 1).any():
        return None
    inv = 1 / o
    return (inv / inv.sum()).tolist()


def _norm(name: str) -> str:
    n = name.lower().replace("&", "and")
    n = re.sub(r"\b(afc|fc)\b", "", n)
    return re.sub(r"[^a-z]", "", n)


def map_team(name: str, known: set[str]) -> str | None:
    if name in known:
        return name
    if TEAM_NAMES.get(name) in known:
        return TEAM_NAMES[name]
    target = _norm(name)
    for k in known:
        if _norm(k) == target:
            return k
    return None


# ------------------------------------------------------------------ The Odds API
CREDITS = {"remaining": None, "used": None}


def fetch_odds_api(key: str, cache: Path) -> list:
    import requests

    events = []
    for region in ("uk", "eu"):
        r = requests.get("https://api.the-odds-api.com/v4/sports/soccer_epl/odds/",
                         params={"apiKey": key, "regions": region, "markets": "h2h", "oddsFormat": "decimal"},
                         timeout=60)
        r.raise_for_status()
        print(f"The Odds API {region}: {len(r.json())} events, credits left {r.headers.get('x-requests-remaining')}")
        CREDITS["remaining"] = r.headers.get("x-requests-remaining")
        CREDITS["used"] = r.headers.get("x-requests-used")
        events.extend(r.json())
    cache.write_text(json.dumps(events))
    return events


def parse_odds_api(events: list, known: set[str]) -> dict:
    """{(home, away): {book_key: {"title": str, "odds": [h, d, a]}}} with football-data team names."""
    out: dict = {}
    unmatched = set()
    for ev in events:
        h, a = map_team(ev["home_team"], known), map_team(ev["away_team"], known)
        if h is None or a is None:
            unmatched |= {ev["home_team"], ev["away_team"]} - set(TEAM_NAMES) - known
            continue
        books = out.setdefault((h, a), {})
        for b in ev.get("bookmakers", []):
            for mk in b.get("markets", []):
                if mk.get("key") != "h2h":
                    continue
                prices = {o["name"]: o["price"] for o in mk.get("outcomes", [])}
                trio = [prices.get(ev["home_team"]), prices.get("Draw"), prices.get(ev["away_team"])]
                if all(isinstance(x, (int, float)) and x > 1 for x in trio):
                    books[b["key"]] = {"title": b.get("title", b["key"]), "odds": [float(x) for x in trio]}
    if unmatched:
        print(f"The Odds API: could not match team names {sorted(unmatched)}")
    return out


# ------------------------------------------------------------------ Opta
def load_opta(path: Path) -> dict:
    """{(home, away, 'YYYY-MM-DD'): [p_home, p_draw, p_away]} from opta.csv (percentages)."""
    if not path.exists():
        return {}
    d = pd.read_csv(path)
    out = {}
    for _, r in d.iterrows():
        h, dr, a = (pd.to_numeric(r.get(c), errors="coerce") for c in ("OptaH", "OptaD", "OptaA"))
        if np.isnan(dr) and not (np.isnan(h) or np.isnan(a)):
            dr = 100 - h - a  # Opta's preview sometimes gives only the two win chances
        if any(np.isnan(x) for x in (h, dr, a)):
            continue
        tot = h + dr + a
        day = pd.to_datetime(r["Date"], dayfirst=True).strftime("%Y-%m-%d")
        out[(str(r["HomeTeam"]).strip(), str(r["AwayTeam"]).strip(), day)] = [h / tot, dr / tot, a / tot]
    return out


# ------------------------------------------------------------------ per-fixture view
def fixture_markets(row: pd.Series, api_books: dict | None, model_p: list, opta_p: list | None,
                    threshold: float) -> dict:
    """Every source's probabilities for one fixture, the sharp consensus, and value bets."""
    sources = []  # (label, probabilities, odds, kind)

    def fd(prefix):
        return [row.get(f"{prefix}{s}") for s in FD_SUFFIX]

    sources.append({"label": "EdgeLab model", "p": model_p, "odds": None, "kind": "model"})
    if opta_p:
        sources.append({"label": "Opta supercomputer", "p": opta_p, "odds": None, "kind": "model"})

    sharp = []  # no-margin probabilities from sharp sources
    api_books = api_books or {}
    pin = api_books.get("pinnacle")
    bfe_live = api_books.get("betfair_ex_uk") or api_books.get("betfair_ex_eu")
    bfe = bfe_live["odds"] if bfe_live else fd("BFE")
    for label, odds in (("Pinnacle", pin["odds"] if pin else None), ("Betfair Exchange", bfe)):
        p = no_vig(odds) if odds is not None else None
        if p:
            sources.append({"label": label, "p": p, "odds": [float(x) for x in odds], "kind": "sharp"})
            sharp.append(p)
    for prefix, label in (("B365", "Bet365"),):
        p = no_vig(fd(prefix))
        if p:
            sources.append({"label": label, "p": p, "odds": [float(x) for x in fd(prefix)], "kind": "book"})
    sb = api_books.get("betfair_sb_uk")
    if sb and no_vig(sb["odds"]):
        sources.append({"label": "Betfair Sportsbook", "p": no_vig(sb["odds"]), "odds": sb["odds"], "kind": "book"})
    p = no_vig(fd("Avg"))
    if p:
        sources.append({"label": "Market average", "p": p, "odds": [float(x) for x in fd("Avg")], "kind": "book"})

    p_sharp = np.mean(sharp, axis=0).tolist() if sharp else None

    # Best bettable price per outcome: football-data books and The Odds API bookmakers.
    offers = []  # (outcome index, price, bookmaker)
    for prefix, name in FD_BOOKS.items():
        for k, x in enumerate(fd(prefix)):
            if isinstance(x, (int, float)) and np.isfinite(x) and x > 1:
                offers.append((k, float(x), name))
    for key, b in api_books.items():
        if key in ODDS_API_SHARP or key in ODDS_API_EXCLUDE or key.startswith("betfair_ex"):
            continue
        for k, x in enumerate(b["odds"]):
            offers.append((k, float(x), b["title"]))
    best = []
    for k in range(3):
        cand = [o for o in offers if o[0] == k]
        if cand:
            top = max(c[1] for c in cand)
            names = sorted({c[2] for c in cand if c[1] == top})
            named = [n for n in names if n != "Best market price"] or names
            best.append({"price": top, "books": named})
        else:
            best.append(None)

    value = []
    if p_sharp:
        for k in range(3):
            if best[k] is None:
                continue
            ev = p_sharp[k] * best[k]["price"] - 1
            if ev >= threshold:
                kelly = max(0.0, ev / (best[k]["price"] - 1))
                value.append({"outcome": OUTCOMES[k], "k": k, "price": best[k]["price"], "books": best[k]["books"],
                              "fair_odds": 1 / p_sharp[k], "p_sharp": p_sharp[k], "ev": ev,
                              "quarter_kelly": kelly / 4, "p_model": model_p[k]})
    return {"sources": sources, "p_sharp": p_sharp, "best": best, "value": value}


# ------------------------------------------------------------------ all bookmakers
FD_DISPLAY = {"B365": "Bet365", "BFD": "Betfred", "BV": "BetVictor", "BW": "Bet&Win", "PP": "Paddy Power",
              "SKB": "Sky Bet", "WH": "William Hill", "LB": "Ladbrokes", "BFE": "Betfair Exchange", "PS": "Pinnacle"}
EXCHANGES = {"Betfair Exchange", "Betfair", "Matchbook", "Smarkets"}


def book_key(name: str) -> str:
    n = re.sub(r"[^a-z0-9]", "", name.lower())
    return n[:-2] if n.endswith("uk") and len(n) > 4 else n


def margin(odds) -> float | None:
    o = np.asarray(odds, float)
    if o.shape != (3,) or not np.isfinite(o).all() or (o <= 1).any():
        return None
    return float((1 / o).sum() - 1)


def fixture_books(row: pd.Series, api_books: dict | None) -> list:
    """Every bookmaker's 1X2 prices for one fixture. Live Odds API prices replace the
    Friday football-data price for the same bookmaker."""
    books: dict = {}
    for prefix, name in FD_DISPLAY.items():
        odds = [row.get(f"{prefix}{x}") for x in FD_SUFFIX]
        if margin(odds) is not None:
            books[book_key(name)] = {"name": name, "odds": [float(x) for x in odds], "source": "Friday"}
    for key, b in (api_books or {}).items():
        name = b["title"]
        if margin(b["odds"]) is not None:
            books[book_key(name)] = {"name": name, "odds": b["odds"], "source": "live"}
    out = []
    for b in books.values():
        mg = margin(b["odds"])
        out.append({**b, "margin": mg, "payout": 1 / (1 + mg),
                    "exchange": b["name"] in EXCHANGES or "exchange" in b["name"].lower()})
    bettable = [b for b in out if not b["exchange"]]
    for k in range(3):
        top = max((b["odds"][k] for b in bettable), default=None)
        for b in out:
            b.setdefault("best", [False, False, False])
            b["best"][k] = (not b["exchange"]) and top is not None and b["odds"][k] >= top - 1e-9
    return sorted(out, key=lambda b: b["margin"])
