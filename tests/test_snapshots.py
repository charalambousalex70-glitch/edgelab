import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgelab import moves  # noqa: E402
from edgelab import schedule as sched  # noqa: E402
from edgelab import snapshots as snaps  # noqa: E402

CFG = {"odds_api_regions": ["uk", "eu"], "odds_api_reset_day": 1}
NOW = datetime(2026, 10, 14, 12, 7, tzinfo=timezone.utc)


def state(**kw):
    base = {"remaining": 400, "fetches": [{"at": "2026-10-14T06:07Z", "mode": "full"}, {"at": "2026-10-14T11:00Z", "mode": "focus"}],
            "kickoffs": ["2026-10-17T14:00Z"], "milestones_done": [], "last_build": "2026-10-14T11:00Z"}
    return {**base, **kw}


def test_first_run_takes_a_full_snapshot():
    assert sched.decide({}, NOW, CFG)[0] == "full"


def test_kickoff_milestones_come_first_and_are_not_repeated():
    st = state(kickoffs=["2026-10-14T13:30Z"])
    mode, reason = sched.decide(st, NOW, CFG)  # 83 minutes before kick-off: the 90-minute snapshot is due
    assert mode == "focus" and "|90" in reason
    st["milestones_done"] = sched.due_milestones(st, NOW, CFG)
    assert sched.decide(st, NOW, CFG)[0] == "skip"


def test_full_snapshot_every_12_hours_and_focus_spacing_follows_the_budget():
    assert sched.decide(state(fetches=[{"at": "2026-10-13T23:00Z", "mode": "full"}]), NOW, CFG)[0] == "full"
    rich, poor = sched.plan(state(remaining=480), NOW, CFG), sched.plan(state(remaining=150), NOW, CFG)
    assert rich["gap_h"] < poor["gap_h"]
    assert sched.decide(state(remaining=480, fetches=[{"at": "2026-10-14T06:07Z", "mode": "full"},
                                                      {"at": "2026-10-14T09:30Z", "mode": "focus"}]), NOW, CFG)[0] == "focus"


def test_runs_out_gracefully_and_rebuilds_without_credits():
    st = state(remaining=1, last_build="2026-10-14T03:00Z")
    assert sched.decide(st, NOW, CFG)[0] == "build"
    assert sched.decide(state(remaining=1), NOW, CFG)[0] == "skip"
    assert sched.decide(state(remaining=1), NOW, CFG, event="push")[0] == "build"


def test_next_reset_is_in_the_future():
    assert sched.next_reset(NOW, 1) == datetime(2026, 11, 1, tzinfo=timezone.utc)
    assert sched.next_reset(NOW, 20) == datetime(2026, 10, 20, tzinfo=timezone.utc)


def _snap(rows):
    df = pd.DataFrame(rows, columns=snaps.COLUMNS)
    for c in ("fetched", "commence", "last_update"):
        df[c] = pd.to_datetime(df[c])
    return df


def test_slow_bookmaker_is_found_and_flagged_on_the_match():
    kick = "2026-10-17 14:00"
    rows = []
    # The sharp home price shortens from 2.50 to 2.00 between 08:00 and 10:00. Fast copies it at once;
    # Slow is still at 2.40 two hours later.
    path = {"08:00": (2.5, 2.5, 2.4), "10:00": (2.0, 2.0, 2.4), "12:00": (2.0, 2.0, 2.4), "14:00": (2.0, 2.0, 2.0)}
    for t, (sharp, fast, slow) in path.items():
        ts = f"2026-10-14 {t}"
        for key, title, home in (("pinnacle", "Pinnacle", sharp), ("betfair_ex_uk", "Betfair", sharp),
                                 ("fastbet", "Fast", fast), ("slowbet", "Slow", slow)):
            rows.append([ts, "focus", kick, "Arsenal", "Leeds", key, title, ts, home, 3.6, 3.6])
    snap = _snap(rows)
    out = moves.analyse(snap[snap.fetched <= "2026-10-14 12:00"], pd.Timestamp("2026-10-14 12:30"), 0.02)
    books = {b["name"]: b for b in out["books"]}
    assert books["Fast"]["follow"] > 0.9 and books["Slow"]["follow"] < 0.1
    card = out["matches"][0]
    assert card["k12"] == 0 and card["lagging"][0]["book"] == "Slow" and card["lagging"][0]["ev"] > 0.1
    assert "Slow" in moves.context(out["matches"], "Arsenal", "Leeds", 0, ["Slow"])
    # once the slow bookmaker follows (14:00), its catch-up time is measured
    out = moves.analyse(snap, pd.Timestamp("2026-10-14 14:30"), 0.02)
    books = {b["name"]: b for b in out["books"]}
    assert books["Slow"]["catchup_h"] > books["Fast"]["catchup_h"]


def test_rows_from_events_maps_teams_and_keeps_last_update(tmp_path):
    ev = {"id": "x", "home_team": "Arsenal", "away_team": "Leeds United", "commence_time": "2026-10-17T14:00:00Z",
          "bookmakers": [{"key": "williamhill", "title": "William Hill", "markets": [{"key": "h2h", "last_update": "2026-10-14T09:12:30Z",
                          "outcomes": [{"name": "Arsenal", "price": 1.5}, {"name": "Draw", "price": 4.2}, {"name": "Leeds United", "price": 7.0}]}]}]}
    t = pd.Timestamp("2026-10-14 09:22")
    df = snaps.rows_from_events([ev, ev], {"Arsenal", "Leeds"}, t, "full")
    assert len(df) == 1 and df.iloc[0][["home", "away", "h", "a"]].tolist() == ["Arsenal", "Leeds", 1.5, 7.0]
    snaps.save(df, tmp_path, t, "full")
    back = snaps.load(tmp_path)
    assert back.last_update.iloc[0] == pd.Timestamp("2026-10-14 09:12:30")
    books = snaps.latest_books(back, pd.Timestamp("2026-10-14 10:00"))
    assert books[("Arsenal", "Leeds")]["williamhill"]["odds"] == [1.5, 4.2, 7.0]


def test_focus_keys_map_titles_from_the_last_full_snapshot():
    st = {"book_keys": {"William Hill": "williamhill", "Betfred (UK)": "betfred_uk", "Pinnacle": "pinnacle"}}
    keys = snaps.focus_keys(st, {"focus_books": ["William Hill", "Betfred", "Nobody"]})
    assert keys == ["pinnacle", "betfair_ex_uk", "williamhill", "betfred_uk"]
