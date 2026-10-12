import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgelab import markets as mk  # noqa: E402

KNOWN = {"Arsenal", "Leeds", "Man City", "Liverpool", "Nott'm Forest", "Brighton", "Bournemouth", "Coventry"}


def test_no_vig_sums_to_one_and_rejects_bad_odds():
    p = mk.no_vig([2.0, 3.5, 4.0])
    assert math.isclose(sum(p), 1.0)
    assert p[0] > p[1] > p[2]
    assert mk.no_vig([2.0, None, 4.0]) is None
    assert mk.no_vig([1.0, 3.5, 4.0]) is None


def test_team_names_map_to_football_data_spelling():
    assert mk.map_team("Manchester City", KNOWN) == "Man City"
    assert mk.map_team("Nottingham Forest", KNOWN) == "Nott'm Forest"
    assert mk.map_team("Brighton and Hove Albion", KNOWN) == "Brighton"
    assert mk.map_team("AFC Bournemouth", KNOWN) == "Bournemouth"
    assert mk.map_team("Leeds United", KNOWN) == "Leeds"
    assert mk.map_team("Some Other FC", KNOWN) is None


def _event(home, away, books):
    return {"home_team": home, "away_team": away, "commence_time": "2026-10-17T14:00:00Z",
            "bookmakers": [{"key": k, "title": t, "markets": [{"key": "h2h", "outcomes": [
                {"name": home, "price": h}, {"name": "Draw", "price": d}, {"name": away, "price": a}]}]}
                for k, t, (h, d, a) in books]}


def test_parse_odds_api_orders_outcomes_home_draw_away():
    ev = _event("Liverpool", "Manchester City", [("pinnacle", "Pinnacle", (2.6, 3.7, 2.6)),
                                                ("williamhill", "William Hill", (2.5, 3.6, 2.7))])
    out = mk.parse_odds_api([ev], KNOWN)
    assert out[("Liverpool", "Man City")]["pinnacle"]["odds"] == [2.6, 3.7, 2.6]
    assert out[("Liverpool", "Man City")]["williamhill"]["title"] == "William Hill"


def _row(**cols):
    return pd.Series(cols, dtype=object)


def test_value_bet_found_when_book_beats_sharp_fair_price():
    # Exchange says home ~50%; a bookmaker offers 2.20 on the home win (EV ~ +9%).
    row = _row(BFEH=2.0, BFED=3.6, BFEA=4.2, B365H=1.95, B365D=3.4, B365A=3.9, MaxH=2.2, MaxD=3.5, MaxA=4.0)
    out = mk.fixture_markets(row, None, [0.5, 0.25, 0.25], None, 0.02)
    homes = [v for v in out["value"] if v["outcome"] == "home"]
    assert homes and homes[0]["price"] == 2.2 and homes[0]["ev"] > 0.05
    assert all(v["outcome"] != "away" for v in out["value"])  # 4.0 is below the fair price


def test_no_value_without_a_sharp_price_and_exchanges_are_never_the_bet():
    row = _row(B365H=2.0, B365D=3.4, B365A=3.9, MaxH=2.5, MaxD=3.5, MaxA=4.0)
    assert mk.fixture_markets(row, None, [0.5, 0.25, 0.25], None, 0.02)["value"] == []
    api = {"pinnacle": {"title": "Pinnacle", "odds": [2.0, 3.6, 4.2]},
           "betfair_ex_uk": {"title": "Betfair", "odds": [2.3, 3.6, 4.2]}}
    out = mk.fixture_markets(_row(), api, [0.5, 0.25, 0.25], None, 0.02)
    assert out["value"] == []  # the only high price is the exchange itself


def test_opta_file_fills_missing_draw(tmp_path):
    f = tmp_path / "opta.csv"
    f.write_text("Date,HomeTeam,AwayTeam,OptaH,OptaD,OptaA\n10/10/2026,Arsenal,Leeds,68.9,,13.7\n")
    p = mk.load_opta(f)[("Arsenal", "Leeds", "2026-10-10")]
    assert p == pytest.approx([0.689, 0.174, 0.137], abs=1e-6)
