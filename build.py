"""Build the EdgeLab forecasts site.

    python build.py                  # download the latest data from football-data.co.uk
    python build.py --local DIR      # use CSVs already in DIR (named E0_<code>.csv)

Steps: load every season in config.json, fit Dixon-Coles (Step 0 model) on all results
so far, predict the upcoming fixtures, rebuild this season's track record (each match
predicted only from earlier results), log the live predictions to history/, and write
public/index.html.
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from edgelab import dixon_coles as dc
from edgelab import markets as mk

ROOT = Path(__file__).parent
DATA, PUBLIC, HISTORY = ROOT / "data", ROOT / "public", ROOT / "history" / "predictions.csv"
VALUE_LOG = ROOT / "history" / "value_bets.csv"
ODDS_LOG = ROOT / "history" / "odds_log.csv"
CFG: dict = {}
OPTA = ROOT / "opta.csv"
# Every odds column we keep: pre-match prices, and closing prices ("C") for settling value bets.
ODDS_PREFIXES = list(mk.FD_BOOKS) + list(mk.FD_SHARP) + list(mk.FD_OTHER)
ODDS_COLS = [f"{b}{x}" for b in ODDS_PREFIXES for x in mk.FD_SUFFIX]
CLOSE_COLS = [f"{b}C{x}" for b in ("BFE", "Avg", "PS", "B365") for x in mk.FD_SUFFIX]
BASE_URL = "https://www.football-data.co.uk"
HEADERS = {"User-Agent": "Mozilla/5.0 (EdgeLab forecasts; weekly build)"}


# ------------------------------------------------------------------ data
def read_csv_text(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.StringIO(raw.decode(enc)))
        except UnicodeDecodeError:
            continue
    raise ValueError("Unreadable CSV")


def download(cfg) -> None:
    import requests

    DATA.mkdir(exist_ok=True)
    for s in cfg["seasons"]:
        url = f"{BASE_URL}/mmz4281/{s['code']}/{cfg['division']}.csv"
        r = requests.get(url, headers=HEADERS, timeout=60)
        if r.status_code == 404 and s["code"] == cfg["current_season"]:
            print(f"No file yet for {s['label']} (season not started)")
            continue
        r.raise_for_status()
        (DATA / f"{cfg['division']}_{s['code']}.csv").write_bytes(r.content)
        print(f"Downloaded {s['label']}")
    try:
        r = requests.get(f"{BASE_URL}/fixtures.csv", headers=HEADERS, timeout=60)
        r.raise_for_status()
        (DATA / "fixtures.csv").write_bytes(r.content)
        print("Downloaded fixtures.csv")
    except Exception as exc:  # the site still builds from fixtures_manual.csv
        print(f"Could not download fixtures.csv ({exc})")


def load_results(cfg) -> pd.DataFrame:
    frames = []
    for s in cfg["seasons"]:
        path = DATA / f"{cfg['division']}_{s['code']}.csv"
        if not path.exists():
            continue
        d = read_csv_text(path.read_bytes())
        d = d.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        time = d["Time"].fillna("15:00") if "Time" in d else "15:00"
        base = pd.DataFrame({
            "date": pd.to_datetime(d["Date"] + " " + time, dayfirst=True),
            "home": d["HomeTeam"].str.strip(), "away": d["AwayTeam"].str.strip(),
            "home_goals": d["FTHG"].astype(int), "away_goals": d["FTAG"].astype(int), "season": s["code"],
        })
        odds = pd.DataFrame({c: pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
                             for c in ODDS_COLS + CLOSE_COLS}, index=d.index)
        frames.append(pd.concat([base, odds], axis=1))
    m = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    m["res"] = np.where(m.home_goals > m.away_goals, "H", np.where(m.home_goals == m.away_goals, "D", "A"))
    return m


def load_fixtures(cfg, results: pd.DataFrame, now: pd.Timestamp) -> tuple[pd.DataFrame, str]:
    played = set(zip(results.home, results.away, results.date.dt.date))

    def tidy(d: pd.DataFrame) -> pd.DataFrame:
        if "Div" in d:
            d = d[d["Div"] == cfg["division"]]
        if d.empty:
            return d
        time = d["Time"].fillna("15:00") if "Time" in d else "15:00"
        out = pd.DataFrame({"date": pd.to_datetime(d["Date"] + " " + time, dayfirst=True),
                            "home": d["HomeTeam"].str.strip(), "away": d["AwayTeam"].str.strip()})
        extra = pd.DataFrame({c: pd.to_numeric(d[c], errors="coerce").to_numpy() if c in d else np.nan
                              for c in ODDS_COLS}, index=out.index)
        out = pd.concat([out, extra], axis=1)
        keep = [(h, a, dt.date()) not in played and dt >= now - timedelta(hours=3)
                for h, a, dt in zip(out.home, out.away, out.date)]
        return out[keep].sort_values("date").reset_index(drop=True)

    fd = tidy(read_csv_text((DATA / "fixtures.csv").read_bytes())) if (DATA / "fixtures.csv").exists() else pd.DataFrame()
    manual_path = ROOT / "fixtures_manual.csv"
    manual = tidy(pd.read_csv(manual_path)) if manual_path.exists() else pd.DataFrame()
    # Hand-typed fixtures always show; football-data's row (with odds) wins for the same match.
    if len(fd) and len(manual):
        have = set(zip(fd.home, fd.away))
        manual = manual[[k not in have for k in zip(manual.home, manual.away)]]
    parts = [x for x in (fd, manual) if len(x)]
    if not parts:
        return pd.DataFrame(columns=["date", "home", "away"]), "none"
    fx = pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)
    source = ("football-data.co.uk (with bookmaker odds)" if len(fd) and not len(manual) else
              "fixtures_manual.csv (odds arrive Friday)" if not len(fd) else
              "football-data.co.uk and fixtures_manual.csv")
    return fx, source


# ------------------------------------------------------------------ model
def promoted_prior(cfg, m, xi):
    las, lbs = [], []
    for s in cfg["seasons"]:
        if s["code"] == cfg["current_season"] or pd.Timestamp(s["end"]) > m.date.max():
            continue
        f = dc.fit(m, as_of=s["end"], xi=xi)
        for t in s["promoted"]:
            if t in f.attack:
                las.append(np.log(f.attack[t])); lbs.append(np.log(f.defence[t]))
    return (float(np.mean(las)), float(np.mean(lbs))) if las else (0.0, 0.0)


def fair_market(row, prefix: str = "Avg") -> list | None:
    p = mk.no_vig([row.get(f"{prefix}{x}") for x in mk.FD_SUFFIX])
    return None if p is None else [round(x, 4) for x in p]


def r4(x):
    return round(float(x), 4)


# ------------------------------------------------------------------ build
def build(cfg, now: pd.Timestamp) -> dict:
    xi, pw = cfg["xi"], cfg["prior_weight"]
    m = load_results(cfg)
    cur = next(s for s in cfg["seasons"] if s["code"] == cfg["current_season"])
    prior = promoted_prior(cfg, m, xi)
    priors = {t: prior for t in cur["promoted"]}

    fit = dc.fit(m, xi=xi, priors=priors, prior_weight=pw)
    week_ago = dc.fit(m, xi=xi, as_of=now - timedelta(days=7), priors=priors, prior_weight=pw)

    teams_now = sorted(set(m[m.season == cur["code"]].home) | set(cur["promoted"]))
    def table(f):
        t = pd.DataFrame([{"team": t, "attack": f._rating(t)[0], "defence": f._rating(t)[1]} for t in teams_now])
        t["net"] = np.log(t.attack) - np.log(t.defence)
        return t.sort_values("net", ascending=False).reset_index(drop=True)
    tab, prev = table(fit), table(week_ago)
    prev_rank = {t: i + 1 for i, t in enumerate(prev.team)}
    ratings = [{"team": r.team, "attack": r4(r.attack), "defence": r4(r.defence), "net": r4(r.net),
                "rank": i + 1, "prev": prev_rank.get(r.team, i + 1)} for i, r in tab.iterrows()]

    fx, fx_source = load_fixtures(cfg, m, now)
    unknown = sorted((set(fx.home) | set(fx.away)) - set(teams_now))
    if unknown:
        raise SystemExit(f"Unknown team names in fixtures (fix spelling to match football-data): {unknown}")
    opta = mk.load_opta(OPTA)
    events = load_odds_api()
    api_books = mk.parse_odds_api(events, set(teams_now)) if events else {}
    fx, fx_source = add_api_fixtures(fx, fx_source, events, set(teams_now), now, m)
    threshold = cfg.get("value_threshold", 0.02)
    fixtures = []
    for _, g in fx.iterrows():
        p = fit.predict(g.home, g.away)
        M = fit.score_matrix(g.home, g.away)
        n = M.shape[0]  # 0..10 goals each side, the same range as the model's score matrix
        top = sorted(((M[i, j], i, j) for i in range(n) for j in range(n)), reverse=True)[:5]
        fixtures.append({
            "date": g.date.strftime("%Y-%m-%dT%H:%M"), "home": g.home, "away": g.away,
            "p": [r4(p["home"]), r4(p["draw"]), r4(p["away"])],
            "xg": [round(p["home_xg"], 2), round(p["away_xg"], 2)],
            "over25": r4(p["over_2_5"]), "btts": r4(p["btts"]),
            "scores": [[int(i), int(j), r4(q)] for q, i, j in top],
            "grid": [[float(f"{M[i, j]:.3g}") for j in range(n)] for i in range(n)],
            "market": fair_market(g),
            "odds": None if fair_market(g) is None else [r4(g.AvgH), r4(g.AvgD), r4(g.AvgA)],
        })
        f = fixtures[-1]
        mkts = mk.fixture_markets(g, api_books.get((g.home, g.away)), f["p"],
                                  opta.get((g.home, g.away, g.date.strftime("%Y-%m-%d"))), threshold)
        f["sources"] = [{**x, "p": [r4(v) for v in x["p"]], "odds": None if x["odds"] is None else [r4(v) for v in x["odds"]]}
                        for x in mkts["sources"]]
        f["sharp"] = None if mkts["p_sharp"] is None else [r4(v) for v in mkts["p_sharp"]]
        f["best"] = mkts["best"]
        f["books"] = [{**b, "odds": [r4(x) for x in b["odds"]], "margin": r4(b["margin"]), "payout": r4(b["payout"])}
                      for b in mk.fixture_books(g, api_books.get((g.home, g.away)))]
        f["value"] = [{**v, **{k: r4(v[k]) for k in ("price", "fair_odds", "p_sharp", "ev", "quarter_kelly", "p_model")}}
                      for v in mkts["value"]]

    # This season's record: each match predicted from results before it only.
    season_m = m[m.season == cur["code"]]
    record = {"matches": []}
    if len(season_m):
        rp = dc.rolling_predictions(m, xi=xi, start=cur["start"], priors=priors, prior_weight=pw)
        rp = rp.merge(m.rename(columns={"home": "home_team", "away": "away_team"}),
                      on=["date", "home_team", "away_team"], suffixes=("", "_m"))
        o = np.stack([rp.res == "H", rp.res == "D", rp.res == "A"], 1)
        pm = rp[["home", "draw", "away"]].to_numpy()
        mkt_series = rp.apply(fair_market, axis=1)
        rp = rp.assign(market=mkt_series.to_numpy())
        has_mk = mkt_series.notna().to_numpy()
        mkt = np.array([x if x is not None else [np.nan] * 3 for x in mkt_series], float)
        ll_model = -np.log((pm * o).sum(1))
        ll_mkt = -np.log(np.where(has_mk, (np.nan_to_num(mkt) * o).sum(1), np.nan))
        fav_model = pm.argmax(1) == o.argmax(1)
        fav_mkt = np.nanargmax(np.nan_to_num(mkt, nan=-1), 1) == o.argmax(1)
        record_sources = source_accuracy(rp, o, pm, opta)
        record = {
            "sources": record_sources,
            "n": int(len(rp)), "n_market": int(has_mk.sum()),
            "logloss_model": r4(ll_model.mean()),
            "logloss_model_same": r4(ll_model[has_mk].mean()) if has_mk.any() else None,
            "logloss_market": r4(np.nanmean(ll_mkt)) if has_mk.any() else None,
            "fav_model": r4(fav_model.mean()), "fav_market": r4(fav_mkt[has_mk].mean()) if has_mk.any() else None,
            "matches": [{
                "date": r.date.strftime("%Y-%m-%d"), "home": r.home_team, "away": r.away_team,
                "score": [int(r.home_goals), int(r.away_goals)],
                "p": [r4(r.home), r4(r.draw), r4(r.away)],
                "market": r.market, "res": r.res,
            } for _, r in rp.sort_values("date", ascending=False).iterrows()],
        }

    live = log_live(fixtures, now, m)
    value = log_value(fixtures, now, m)
    log_odds(fixtures, now)
    board = bookmaker_board(cfg, m, rp if len(season_m) else None)
    return {
        "generated": now.strftime("%Y-%m-%dT%H:%M"), "league": cfg["league"], "season": cur["label"],
        "last_result": m.date.max().strftime("%Y-%m-%d"), "fixtures_source": fx_source,
        "model": {"home": r4(fit.home), "rho": r4(fit.rho), "xi": xi, "matches": fit.n_matches,
                  "promoted_prior": [r4(np.exp(prior[0])), r4(np.exp(prior[1]))]},
        "fixtures": fixtures, "ratings": ratings, "record": record, "live": live, "value": value,
        "board": board,
        "value_threshold": threshold, "odds_api": bool(api_books), "odds_api_credits": mk.CREDITS["remaining"],
        "backtest": cfg.get("backtest"),
    }


def _book_rows(odds: np.ndarray, o: np.ndarray, ll_ref: np.ndarray, best: np.ndarray | None, days) -> dict | None:
    """Payout and accuracy stats for one bookmaker over the matches it priced."""
    ok = np.isfinite(odds).all(1) & (odds > 1).all(1)
    if ok.sum() < 10:
        return None
    inv = 1 / odds[ok]
    tot = inv.sum(1)
    p = inv / tot[:, None]
    ll = -np.log((p * o[ok]).sum(1))
    d = ll - ll_ref[ok]
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    row = {"n": int(ok.sum()), "margin": r4((tot - 1).mean()), "payout": r4((1 / tot).mean()),
           "logloss": r4(ll.mean()), "vs_ref": r4(d.mean()), "vs_ref_ci": r4(1.645 * se) if np.isfinite(se) else None,
           "best_share": None, "fav": r4((p.argmax(1) == o[ok].argmax(1)).mean())}
    if best is not None:
        row["best_share"] = r4(np.mean(np.isclose(odds[ok], best[ok])))
    return row


def bookmaker_board(cfg, m: pd.DataFrame, rp: pd.DataFrame | None) -> dict:
    """Who pays the most (lowest margin, most often the best price) and who is the most
    accurate (log loss of margin-free chances, against the market average on the same
    matches), for this season, last season and every season loaded."""
    o_all = np.stack([m.res == "H", m.res == "D", m.res == "A"], 1).astype(float)
    avg = m[["AvgH", "AvgD", "AvgA"]].to_numpy(float)
    p_avg = (1 / avg) / (1 / avg).sum(1, keepdims=True)
    ll_avg = -np.log((p_avg * o_all).sum(1))
    sportsbooks = [b for b in mk.FD_DISPLAY if b not in ("BFE", "PS")]
    stack = np.stack([m[[f"{b}{x}" for x in mk.FD_SUFFIX]].to_numpy(float) for b in sportsbooks])
    best = np.nanmax(np.where(np.isfinite(stack), stack, -np.inf), 0)
    codes = [s["code"] for s in cfg["seasons"]]
    cur, prev = cfg["current_season"], (codes[codes.index(cfg["current_season"]) - 1] if len(codes) > 1 else None)
    labels = {s["code"]: s["label"] for s in cfg["seasons"]}
    windows = [("season", f"This season ({labels[cur]})", m.season == cur)]
    if prev:
        windows.append(("last", f"Last season ({labels[prev]})", m.season == prev))
    windows.append(("all", f"Since {labels[codes[0]].split('/')[0]}", m.season.notna()))
    out = {"windows": []}
    for key, label, mask in windows:
        mask = mask.to_numpy()
        rows = []
        for prefix, name in list(mk.FD_DISPLAY.items()) + [("Avg", "Market average")]:
            od = m[[f"{prefix}{x}" for x in mk.FD_SUFFIX]].to_numpy(float).copy()
            od[~mask] = np.nan
            r = _book_rows(od, o_all, ll_avg, best if prefix in sportsbooks else None, m.date)
            if r:
                rows.append({"name": name, "exchange": prefix == "BFE", "reference": prefix == "Avg", **r})
        if key == "season" and rp is not None and len(rp):
            pm = rp[["home", "draw", "away"]].to_numpy()
            o = np.stack([rp.res == "H", rp.res == "D", rp.res == "A"], 1).astype(float)
            ra = rp[["AvgH", "AvgD", "AvgA"]].to_numpy(float)
            ok = np.isfinite(ra).all(1)
            if ok.any():
                pa = (1 / ra[ok]) / (1 / ra[ok]).sum(1, keepdims=True)
                ll = -np.log((pm[ok] * o[ok]).sum(1)); d = ll + np.log((pa * o[ok]).sum(1))
                rows.append({"name": "EdgeLab model", "model": True, "n": int(ok.sum()), "margin": None, "payout": None,
                             "logloss": r4(ll.mean()), "vs_ref": r4(d.mean()),
                             "vs_ref_ci": r4(1.645 * d.std(ddof=1) / np.sqrt(len(d))) if ok.sum() > 1 else None,
                             "best_share": None, "fav": r4((pm[ok].argmax(1) == o[ok].argmax(1)).mean())})
        out["windows"].append({"key": key, "label": label, "rows": rows})
    out["live"] = live_board(m)
    return out


def log_odds(fixtures: list, now: pd.Timestamp) -> None:
    """Save the first live snapshot of every bookmaker's prices for each match (all books
    from the same moment, so they can be compared fairly once the result is in)."""
    rows = [{"logged_at": now.strftime("%Y-%m-%d %H:%M"), "date": f["date"], "home": f["home"], "away": f["away"],
             "book": b["name"], "exchange": b["exchange"], "h": b["odds"][0], "d": b["odds"][1], "a": b["odds"][2]}
            for f in fixtures for b in f.get("books", []) if b["source"] == "live"]
    if not rows:
        return
    new = pd.DataFrame(rows)
    hist = pd.read_csv(ODDS_LOG) if ODDS_LOG.exists() else pd.DataFrame()
    if len(hist):
        seen = set(zip(hist.date, hist.home, hist.away))
        new = new[[k not in seen for k in zip(new.date, new.home, new.away)]]
    if len(new):
        pd.concat([hist, new], ignore_index=True).to_csv(ODDS_LOG, index=False)


def live_board(m: pd.DataFrame) -> list:
    """Payout and accuracy for every bookmaker in the live log, on settled matches.
    Accuracy is against the average of all logged bookmakers on the same matches."""
    if not ODDS_LOG.exists():
        return []
    h = pd.read_csv(ODDS_LOG)
    h["d0"] = pd.to_datetime(h.date).dt.date
    res = m.assign(d0=m.date.dt.date)[["d0", "home", "away", "res"]]
    s = h.merge(res, on=["d0", "home", "away"])
    if s.empty:
        return []
    s["key"] = s.date + s.home + s.away
    inv = 1 / s[["h", "d", "a"]].to_numpy(float)
    s[["ph", "pd", "pa"]] = inv / inv.sum(1, keepdims=True)
    s["won_p"] = np.choose(s.res.map({"H": 0, "D": 1, "A": 2}).to_numpy(), s[["ph", "pd", "pa"]].to_numpy().T)
    s["ll"] = -np.log(s.won_p)
    s["ll_ref"] = s.groupby("key").ll.transform("mean")
    s["mg"] = inv.sum(1) - 1
    best = s[~s.exchange.astype(bool)].groupby("key")[["h", "d", "a"]].max()
    rows = []
    for name, g in s.groupby("book"):
        d = (g.ll - g.ll_ref).to_numpy()
        bb = best.loc[g.key]
        share = None if bool(g.exchange.iloc[0]) else r4(np.isclose(g[["h", "d", "a"]].to_numpy(), bb.to_numpy()).mean())
        rows.append({"name": name, "exchange": bool(g.exchange.iloc[0]), "n": int(len(g)), "margin": r4(g.mg.mean()),
                     "payout": r4((1 / (1 + g.mg)).mean()), "logloss": r4(g.ll.mean()), "vs_ref": r4(d.mean()),
                     "vs_ref_ci": r4(1.645 * d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None,
                     "best_share": share})
    return sorted(rows, key=lambda r: r["margin"])


def load_odds_api() -> list | None:
    """Live prices from The Odds API on every run when a key is set (2 credits a run).
    The feed lists matches as soon as bookmakers price them, often a week or more ahead,
    which is when soft prices are most likely."""
    import os

    cache = DATA / "odds_api.json"
    key = os.environ.get("ODDS_API_KEY", "").strip()
    status = ROOT / "history" / "odds_api_status.txt"

    def note(msg: str) -> None:  # committed with the history so problems are visible without logs
        status.write_text(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC: {msg}\n")
        print(f"The Odds API: {msg}")

    if key:
        try:
            events = mk.fetch_odds_api(key, cache, CFG.get("odds_api_regions", ["uk", "eu"]))
            note(f"ok, {len(events)} events across uk+eu, credits left {mk.CREDITS['remaining']}")
            return events
        except Exception as exc:  # never let the optional feed break the site
            resp = getattr(exc, "response", None)
            detail = f"HTTP {resp.status_code}: {resp.text[:200]}" if resp is not None else str(exc)[:200]
            note(f"request failed ({detail}) - key length {len(key)}")
            return None
    if os.environ.get("GITHUB_ACTIONS"):
        note("no ODDS_API_KEY secret visible to this run")
    if cache.exists():
        return json.loads(cache.read_text())  # local runs with a saved payload
    return None


def add_api_fixtures(fx: pd.DataFrame, source: str, events: list | None, known: set, now: pd.Timestamp,
                     results: pd.DataFrame, days_ahead: int = 10) -> tuple[pd.DataFrame, str]:
    """Add upcoming matches that The Odds API already prices but football-data has not
    listed yet, so value can be spotted early in the week."""
    if not events:
        return fx, source
    have = set(zip(fx.home, fx.away)) if len(fx) else set()
    played = set(zip(results.home, results.away, results.season))
    season = results.season.iloc[-1] if len(results) else None
    rows = []
    for ev in events:
        h, a = mk.map_team(ev["home_team"], known), mk.map_team(ev["away_team"], known)
        if h is None or a is None or (h, a) in have or (h, a, season) in played:
            continue
        kick = pd.Timestamp(ev["commence_time"]).tz_convert("Europe/London").tz_localize(None)
        if not (now <= kick <= now + timedelta(days=days_ahead)):
            continue
        have.add((h, a))
        rows.append({"date": kick, "home": h, "away": a})
    if not rows:
        return fx, source
    print(f"Added {len(rows)} fixtures from The Odds API that football-data has not listed yet")
    extra = pd.DataFrame(rows)
    out = pd.concat([fx, extra], ignore_index=True) if len(fx) else extra
    for c in ODDS_COLS:
        if c not in out:
            out[c] = np.nan
    label = source if len(fx) else "The Odds API (live prices)"
    return out.sort_values("date").reset_index(drop=True), label


def source_accuracy(rp: pd.DataFrame, o: np.ndarray, pm: np.ndarray, opta: dict) -> list:
    """Season-to-date log loss for each source, each on the matches where it has prices,
    next to the model's log loss on exactly those matches."""
    out = []
    ll_model = -np.log((pm * o).sum(1))

    def add(label, probs):
        ok = np.array([p is not None for p in probs])
        if not ok.any():
            return
        P = np.array([p for p in probs if p is not None], float)
        ll = -np.log((P * o[ok]).sum(1))
        fav = P.argmax(1) == o[ok].argmax(1)
        out.append({"label": label, "n": int(ok.sum()), "logloss": r4(ll.mean()),
                    "model_same": r4(ll_model[ok].mean()), "fav": r4(fav.mean())})

    out.append({"label": "EdgeLab model", "n": int(len(pm)), "logloss": r4(ll_model.mean()),
                "model_same": r4(ll_model.mean()), "fav": r4((pm.argmax(1) == o.argmax(1)).mean())})
    add("Opta supercomputer", [opta.get((r.home_team, r.away_team, r.date.strftime("%Y-%m-%d"))) for _, r in rp.iterrows()])
    for prefix, label in (("PS", "Pinnacle"), ("BFE", "Betfair Exchange"), ("B365", "Bet365"), ("Avg", "Market average")):
        add(label, [fair_market(r, prefix) for _, r in rp.iterrows()])
    return out


def write_alert(new: pd.DataFrame) -> None:
    """data/alert.md lists value bets found for the first time this run; the workflow turns
    it into a GitHub issue, which GitHub sends to the owner by email and to its mobile app."""
    alert = DATA / "alert.md"
    alert.unlink(missing_ok=True)
    if new is None or not len(new):
        return
    names = {"home": None, "draw": "Draw", "away": None}
    lines = ["New value bets found by EdgeLab (prices beat the betting exchange's fair price):", ""]
    for _, r in new.iterrows():
        pick = names[r.outcome] or f"{r.home if r.outcome == 'home' else r.away} to win"
        lines.append(f"- **{pick}** ({r.home} v {r.away}, {pd.Timestamp(r.date):%a %d %b %H:%M} UK): "
                     f"{r.price:.2f} with {r.books}, fair price {1 / r.p_sharp:.2f}, value +{r.ev * 100:.1f}%")
    lines += ["", "Prices move fast; check the bookmaker before betting. Full details on the site.",
              "18+. Only bet what you can afford to lose."]
    DATA.mkdir(exist_ok=True)
    alert.write_text("\n".join(lines) + "\n")
    (DATA / "alert_title.txt").write_text(f"{len(new)} new value bet{'s' if len(new) != 1 else ''}")


def log_value(fixtures: list, now: pd.Timestamp, m: pd.DataFrame) -> dict:
    """Log each value bet the first time it appears, then settle it against the result and
    the closing exchange price (closing line value)."""
    rows = [{"logged_at": now.strftime("%Y-%m-%d %H:%M"), "date": f["date"], "home": f["home"], "away": f["away"],
             "outcome": v["outcome"], "price": v["price"], "books": " / ".join(v["books"]),
             "p_sharp": v["p_sharp"], "ev": v["ev"], "p_model": v["p_model"]}
            for f in fixtures for v in f.get("value", [])]
    hist = pd.read_csv(VALUE_LOG) if VALUE_LOG.exists() else pd.DataFrame()
    new = pd.DataFrame(rows)
    if len(hist) and len(new):
        seen = set(zip(hist.date, hist.home, hist.away, hist.outcome))
        new = new[[k not in seen for k in zip(new.date, new.home, new.away, new.outcome)]]
    hist = pd.concat([hist, new], ignore_index=True)
    if len(hist):
        hist.to_csv(VALUE_LOG, index=False)
    write_alert(new)
    out = {"logged": int(len(hist)), "settled": 0, "bets": []}
    if hist.empty:
        return out
    hist["d"] = pd.to_datetime(hist.date).dt.date
    res = m.assign(d=m.date.dt.date)
    s = hist.merge(res, on=["d", "home", "away"], how="left", suffixes=("", "_r"))
    k = s.outcome.map({"home": 0, "draw": 1, "away": 2}).to_numpy()
    won = (s.res.map({"H": 0, "D": 1, "A": 2}).to_numpy() == k)
    settled = s.res.notna().to_numpy()

    def closing(r, j):
        for prefix in ("BFEC", "AvgC"):
            p = mk.no_vig([r.get(f"{prefix}{x}") for x in mk.FD_SUFFIX])
            if p:
                return p[j]
        return np.nan
    p_close = np.array([closing(r, j) if ok else np.nan for (_, r), j, ok in zip(s.iterrows(), k, settled)])
    pl = np.where(won, s.price - 1, -1.0)
    clv = p_close * s.price - 1
    sel = settled
    out["settled"] = int(sel.sum())
    if sel.any():
        c = clv[sel][np.isfinite(clv[sel])]
        se = c.std(ddof=1) / np.sqrt(len(c)) if len(c) > 1 else np.nan
        out.update({"profit": r4(pl[sel].sum()), "roi": r4(pl[sel].mean()), "wins": int(won[sel].sum()),
                    "clv": r4(c.mean()) if len(c) else None,
                    "clv_lo": r4(c.mean() - 1.645 * se) if len(c) > 1 else None,
                    "clv_hi": r4(c.mean() + 1.645 * se) if len(c) > 1 else None})
    order = np.argsort(pd.to_datetime(s.date).to_numpy())[::-1]
    out["bets"] = [{"date": s.date.iloc[i][:10], "home": s.home.iloc[i], "away": s.away.iloc[i],
                    "outcome": s.outcome.iloc[i], "price": r4(s.price.iloc[i]), "books": s.books.iloc[i],
                    "ev": r4(s.ev.iloc[i]), "settled": bool(settled[i]), "won": bool(won[i]) if settled[i] else None,
                    "clv": None if not np.isfinite(clv[i]) else r4(clv[i])} for i in order[:60]]
    return out


def log_live(fixtures, now, m) -> dict:
    """Append this run's predictions; per match keep the earliest one made with odds."""
    HISTORY.parent.mkdir(exist_ok=True)
    def src(f, label):
        p = next((x["p"] for x in f.get("sources", []) if x["label"] == label), None)
        return p or [None] * 3

    rows = pd.DataFrame([{"predicted_at": now.strftime("%Y-%m-%d %H:%M"), "date": f["date"],
                          "home": f["home"], "away": f["away"], "p_home": f["p"][0], "p_draw": f["p"][1],
                          "p_away": f["p"][2], "has_odds": f["odds"] is not None,
                          "odds_home": (f["odds"] or [None] * 3)[0], "odds_draw": (f["odds"] or [None] * 3)[1],
                          "odds_away": (f["odds"] or [None] * 3)[2],
                          **{f"{tag}_{o}": src(f, label)[i] for tag, label in
                             (("pin", "Pinnacle"), ("bfe", "Betfair Exchange"), ("b365", "Bet365"),
                              ("opta", "Opta supercomputer"))
                             for i, o in enumerate(("home", "draw", "away"))}} for f in fixtures])
    hist = pd.read_csv(HISTORY) if HISTORY.exists() else pd.DataFrame()
    if len(hist) and len(rows):
        # Only add a row if the match is new, or if it now has odds and no priced row exists yet.
        key = lambda d: list(zip(d.date, d.home, d.away))
        seen = set(key(hist))
        priced = set(key(hist[hist.has_odds.astype(bool)]))
        rows = rows[[(k not in seen) or (bool(o) and k not in priced) for k, o in zip(key(rows), rows.has_odds)]]
    hist = pd.concat([hist, rows], ignore_index=True)
    if len(hist):
        hist.to_csv(HISTORY, index=False)
    if hist.empty:
        return {"logged": 0, "settled": 0}
    first = (hist.sort_values(["has_odds", "predicted_at"], ascending=[False, True])
                 .drop_duplicates(["date", "home", "away"]))
    first["d"] = pd.to_datetime(first.date).dt.date
    res = m.assign(d=m.date.dt.date)[["d", "home", "away", "res"]]
    s = first.merge(res, on=["d", "home", "away"])
    if s.empty:
        return {"logged": int(len(first)), "settled": 0}
    p = s[["p_home", "p_draw", "p_away"]].to_numpy()
    o = np.stack([s.res == "H", s.res == "D", s.res == "A"], 1)
    return {"logged": int(len(first)), "settled": int(len(s)),
            "logloss": r4(-np.log((p * o).sum(1)).mean()), "fav_hit": r4((p.argmax(1) == o.argmax(1)).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", help="folder with E0_<code>.csv files (skip downloading)")
    ap.add_argument("--now", help="pretend the current time is this (YYYY-MM-DD HH:MM)")
    args = ap.parse_args()
    cfg = json.loads((ROOT / "config.json").read_text())
    CFG.update(cfg)
    now = pd.Timestamp(args.now) if args.now else pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    if args.local:
        DATA.mkdir(exist_ok=True)
        for f in Path(args.local).glob("*.csv"):
            shutil.copy(f, DATA / f.name)
    else:
        download(cfg)
    payload = build(cfg, now)
    PUBLIC.mkdir(exist_ok=True)
    html = (ROOT / "template.html").read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
    (PUBLIC / "index.html").write_text(html, encoding="utf-8")
    print(f"Built public/index.html: {len(payload['fixtures'])} fixtures, "
          f"{payload['record'].get('n', 0)} matches in the record, fixtures from {payload['fixtures_source']}")


if __name__ == "__main__":
    main()
