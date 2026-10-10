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

ROOT = Path(__file__).parent
DATA, PUBLIC, HISTORY = ROOT / "data", ROOT / "public", ROOT / "history" / "predictions.csv"
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
        for c in ["AvgH", "AvgD", "AvgA"]:
            if c not in d:
                d[c] = np.nan
        time = d["Time"].fillna("15:00") if "Time" in d else "15:00"
        frames.append(pd.DataFrame({
            "date": pd.to_datetime(d["Date"] + " " + time, dayfirst=True),
            "home": d["HomeTeam"].str.strip(), "away": d["AwayTeam"].str.strip(),
            "home_goals": d["FTHG"].astype(int), "away_goals": d["FTAG"].astype(int),
            "AvgH": d["AvgH"], "AvgD": d["AvgD"], "AvgA": d["AvgA"], "season": s["code"],
        }))
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
        for c in ["AvgH", "AvgD", "AvgA", "MaxH", "MaxD", "MaxA"]:
            out[c] = pd.to_numeric(d[c], errors="coerce").to_numpy() if c in d else np.nan
        keep = [(h, a, dt.date()) not in played and dt >= now - timedelta(hours=3)
                for h, a, dt in zip(out.home, out.away, out.date)]
        return out[keep].sort_values("date").reset_index(drop=True)

    if (DATA / "fixtures.csv").exists():
        fx = tidy(read_csv_text((DATA / "fixtures.csv").read_bytes()))
        if len(fx):
            return fx, "football-data.co.uk (with bookmaker odds)"
    manual = ROOT / "fixtures_manual.csv"
    if manual.exists():
        fx = tidy(pd.read_csv(manual))
        if len(fx):
            return fx, "fixtures_manual.csv (no odds yet)"
    return pd.DataFrame(columns=["date", "home", "away"]), "none"


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


def fair_market(row) -> list | None:
    o = np.array([row.get("AvgH"), row.get("AvgD"), row.get("AvgA")], float)
    if not np.isfinite(o).all() or (o <= 1).any():
        return None
    inv = 1 / o
    return (inv / inv.sum()).round(4).tolist()


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

    # This season's record: each match predicted from results before it only.
    season_m = m[m.season == cur["code"]]
    record = {"matches": []}
    if len(season_m):
        rp = dc.rolling_predictions(m, xi=xi, start=cur["start"], priors=priors, prior_weight=pw)
        rp = rp.merge(m.rename(columns={"home": "home_team", "away": "away_team"}),
                      on=["date", "home_team", "away_team"], suffixes=("", "_m"))
        o = np.stack([rp.res == "H", rp.res == "D", rp.res == "A"], 1)
        pm = rp[["home", "draw", "away"]].to_numpy()
        mk = rp.apply(fair_market, axis=1)
        rp = rp.assign(market=mk.to_numpy())
        has_mk = mk.notna().to_numpy()
        mkt = np.array([x if x is not None else [np.nan] * 3 for x in mk], float)
        ll_model = -np.log((pm * o).sum(1))
        ll_mkt = -np.log(np.where(has_mk, (np.nan_to_num(mkt) * o).sum(1), np.nan))
        fav_model = pm.argmax(1) == o.argmax(1)
        fav_mkt = np.nanargmax(np.nan_to_num(mkt, nan=-1), 1) == o.argmax(1)
        record = {
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
    return {
        "generated": now.strftime("%Y-%m-%dT%H:%M"), "league": cfg["league"], "season": cur["label"],
        "last_result": m.date.max().strftime("%Y-%m-%d"), "fixtures_source": fx_source,
        "model": {"home": r4(fit.home), "rho": r4(fit.rho), "xi": xi, "matches": fit.n_matches,
                  "promoted_prior": [r4(np.exp(prior[0])), r4(np.exp(prior[1]))]},
        "fixtures": fixtures, "ratings": ratings, "record": record, "live": live,
    }


def log_live(fixtures, now, m) -> dict:
    """Append this run's predictions; per match keep the earliest one made with odds."""
    HISTORY.parent.mkdir(exist_ok=True)
    rows = pd.DataFrame([{"predicted_at": now.strftime("%Y-%m-%d %H:%M"), "date": f["date"],
                          "home": f["home"], "away": f["away"], "p_home": f["p"][0], "p_draw": f["p"][1],
                          "p_away": f["p"][2], "has_odds": f["odds"] is not None,
                          "odds_home": (f["odds"] or [None] * 3)[0], "odds_draw": (f["odds"] or [None] * 3)[1],
                          "odds_away": (f["odds"] or [None] * 3)[2]} for f in fixtures])
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
