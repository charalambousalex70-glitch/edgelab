"""Every Odds API price snapshot, saved for movement and lag analysis.

Each fetch is one small gzipped CSV in history/snapshots/YYYY-MM/, never edited after:
    fetched, mode, commence, home, away, book_key, book, last_update, h, d, a
Times are UTC. `last_update` is when The Odds API last saw that bookmaker change its
1X2 prices, which dates a price change more finely than our snapshot spacing.
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import markets as mk

SHARP_KEYS = ("pinnacle", "betfair_ex_uk", "betfair_ex_eu")
EXCHANGE_KEYS = {"betfair_ex_uk", "betfair_ex_eu", "matchbook", "smarkets"}
COLUMNS = ["fetched", "mode", "commence", "home", "away", "book_key", "book", "last_update", "h", "d", "a"]
URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds/"


def fetch(key: str, mode: str, regions: list[str], focus_keys: list[str]) -> tuple[list, dict]:
    """Download one snapshot. Returns (events, quota headers)."""
    import requests

    calls = [{"bookmakers": ",".join(focus_keys)}] if mode == "focus" else [{"regions": r} for r in regions]
    events, quota, cost = [], {}, 0
    for extra in calls:
        r = requests.get(URL, params={"apiKey": key, "markets": "h2h", "oddsFormat": "decimal", **extra}, timeout=60)
        r.raise_for_status()
        cost += int(float(r.headers.get("x-requests-last") or 0))
        quota = {"remaining": r.headers.get("x-requests-remaining"), "used": r.headers.get("x-requests-used"), "last": cost}
        print(f"The Odds API {extra}: {len(r.json())} events, cost {quota['last']}, credits left {quota['remaining']}")
        events.extend(r.json())
    mk.CREDITS.update(remaining=quota.get("remaining"), used=quota.get("used"))
    return events, quota


def rows_from_events(events: list, known: set[str], fetched: pd.Timestamp, mode: str) -> pd.DataFrame:
    rows, seen = [], set()
    for ev in events:
        h, a = mk.map_team(ev["home_team"], known), mk.map_team(ev["away_team"], known)
        if h is None or a is None:
            continue
        for b in ev.get("bookmakers", []):
            if (ev.get("id"), h, a, b["key"]) in seen:
                continue
            for m in b.get("markets", []):
                if m.get("key") != "h2h":
                    continue
                prices = {o["name"]: o["price"] for o in m.get("outcomes", [])}
                trio = [prices.get(ev["home_team"]), prices.get("Draw"), prices.get(ev["away_team"])]
                if not all(isinstance(x, (int, float)) and x > 1 for x in trio):
                    continue
                seen.add((ev.get("id"), h, a, b["key"]))
                rows.append({"fetched": fetched.strftime("%Y-%m-%dT%H:%MZ"), "mode": mode,
                             "commence": ev["commence_time"].replace(":00Z", "Z"), "home": h, "away": a,
                             "book_key": b["key"], "book": b.get("title", b["key"]),
                             "last_update": (m.get("last_update") or b.get("last_update") or "").replace(":00Z", "Z") or None,
                             "h": float(trio[0]), "d": float(trio[1]), "a": float(trio[2])})
    return pd.DataFrame(rows, columns=COLUMNS)


def save(df: pd.DataFrame, folder: Path, fetched: pd.Timestamp, mode: str) -> Path | None:
    if df.empty:
        return None
    sub = folder / fetched.strftime("%Y-%m")
    sub.mkdir(parents=True, exist_ok=True)
    path = sub / f"{fetched:%Y%m%dT%H%M}Z-{mode}.csv.gz"
    df.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    return path


def load(folder: Path, since: pd.Timestamp | None = None) -> pd.DataFrame:
    """All snapshots (optionally only files fetched on or after `since`), times as naive UTC."""
    files = sorted(folder.glob("*/*.csv.gz"))
    if since is not None:
        stamp = since.strftime("%Y%m%dT%H%M")
        files = [f for f in files if f.name[:13] >= stamp]
    if not files:
        return pd.DataFrame(columns=COLUMNS)
    s = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for c in ("fetched", "commence", "last_update"):
        s[c] = pd.to_datetime(s[c], utc=True, errors="coerce").dt.tz_localize(None)
    return s


def latest_books(s: pd.DataFrame, now: pd.Timestamp, keep_h: float = 26) -> dict:
    """Latest price per match and bookmaker, from snapshots in the last `keep_h` hours, as
    {(home, away): {book_key: {"title", "odds", "fetched", "stale_h"}}}. `stale_h` is how much
    older the bookmaker's price is than the newest sharp price for the same match; the value
    finder ignores prices that are too stale to trust."""
    if s.empty:
        return {}
    r = s[(s.fetched >= now - timedelta(hours=keep_h)) & (s.commence > now)]
    if r.empty:
        return {}
    r = r.sort_values("fetched").drop_duplicates(["home", "away", "book_key"], keep="last")
    out: dict = {}
    for (h, a), g in r.groupby(["home", "away"]):
        sharp_t = g[g.book_key.isin(SHARP_KEYS)].fetched.max()
        books = {}
        for x in g.itertuples():
            stale = 0.0 if pd.isna(sharp_t) else max(0.0, (sharp_t - x.fetched).total_seconds() / 3600)
            books[x.book_key] = {"title": x.book, "odds": [float(x.h), float(x.d), float(x.a)],
                                 "fetched": x.fetched.strftime("%Y-%m-%dT%H:%MZ"), "stale_h": round(stale, 2)}
        out[(h, a)] = books
    return out


def upcoming(s: pd.DataFrame, now: pd.Timestamp, days: int = 14) -> list[dict]:
    """Matches priced in the newest snapshot, with UTC kick-off times."""
    if s.empty:
        return []
    last = s[s.fetched == s.fetched.max()]
    last = last[(last.commence > now) & (last.commence <= now + timedelta(days=days))]
    return [{"home": h, "away": a, "commence": c} for (h, a, c), _ in last.groupby(["home", "away", "commence"])]


def update_state(path: Path, state: dict, now: pd.Timestamp, mode: str | None, quota: dict | None,
                 rows: pd.DataFrame | None, s: pd.DataFrame, due: list[str]) -> dict:
    """Record a fetch (or just a rebuild) for the scheduler."""
    state = dict(state)
    stamp = now.strftime("%Y-%m-%dT%H:%MZ")
    state["last_build"] = stamp
    if mode and quota:
        rem = pd.to_numeric(quota.get("remaining"), errors="coerce")
        prev = state.get("remaining")
        if prev is not None and np.isfinite(rem) and rem > prev + 5:
            state["reset_day"] = now.day  # the quota went back up: this is the reset day
        if np.isfinite(rem):
            state["remaining"] = int(rem)
        fetches = state.get("fetches", [])
        fetches.append({"at": stamp, "mode": mode, "cost": quota.get("last"), "remaining": state.get("remaining"),
                        "rows": 0 if rows is None else int(len(rows))})
        state["fetches"] = fetches[-300:]
        done = [x for x in state.get("milestones_done", []) if pd.Timestamp(x.split("|")[0]).tz_convert(None) > now - timedelta(days=2)]
        state["milestones_done"] = sorted(set(done) | set(due))
        if mode == "full" and rows is not None and len(rows):
            names = dict(state.get("book_keys", {}))
            names.update(dict(zip(rows.book, rows.book_key)))
            state["book_keys"] = dict(sorted(names.items()))
    kick = sorted({c.strftime("%Y-%m-%dT%H:%MZ") for c in s.commence.dropna() if now - timedelta(hours=3) < c <= now + timedelta(days=14)}) if len(s) else []
    if kick:
        state["kickoffs"] = kick
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=1) + "\n")
    return state


def focus_keys(state: dict, cfg: dict) -> list[str]:
    """Pinnacle, the Betfair Exchange and up to eight UK bookmakers named in config.json,
    mapped to API keys from the last full snapshot. Ten bookmakers cost one credit."""
    names = {mk.book_key(t): k for t, k in state.get("book_keys", {}).items()}
    wanted = [names.get(mk.book_key(t)) for t in cfg.get("focus_books", [])]
    keys = ["pinnacle", "betfair_ex_uk"] + [k for k in wanted if k and k not in ("pinnacle", "betfair_ex_uk")]
    return list(dict.fromkeys(keys))[:10]
