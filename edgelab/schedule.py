"""When to spend Odds API credits. Standard library only, so the workflow's gate job can
run it in a second without installing anything.

    python -m edgelab.schedule --event schedule      # prints "mode=full|focus|build|skip"

The free plan gives 500 credits a month. A "full" snapshot asks for the uk and eu regions
(2 credits, every bookmaker the API lists). A "focus" snapshot asks for ten named
bookmakers (1 credit): Pinnacle, the Betfair Exchange and eight UK bookmakers.

Plan, checked every 15 minutes between 05:00 and 23:00 UTC by the workflow:
  1. Kick-off milestones: a focus snapshot 90, 35 and 12 minutes before every kick-off
     time (before team news, just after the line-ups, and as close to the off as GitHub's
     timer allows). This gives the closing price of every bookmaker and catches the
     line-up move, when slow bookmakers are most exposed.
  2. A full snapshot every 12 hours, so every bookmaker is tracked.
  3. Focus snapshots in between, spaced so the month's credits last until the reset:
     spare credits = credits left - planned full snapshots - planned milestones - reserve.
  4. Otherwise rebuild the site (no credits) if it is more than 6 hours old.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "history" / "snapshot_state.json"

DEFAULTS = {
    "full_every_h": 12,
    "milestones_min": [90, 35, 12],
    "min_gap_h": 2.0,
    "max_gap_h": 8.0,
    "quiet_gap_h": 8.0,          # gap when the next kick-off is more than 5 days away
    "reserve": 10,
    "slots_per_day": 0.9,        # distinct Premier League kick-off times per day, on average
    "active_hours": 18,          # the workflow only ticks 05:00-23:00 UTC
    "rebuild_every_h": 6,
}


def settings(cfg: dict) -> dict:
    return {**DEFAULTS, **cfg.get("snapshots", {})}


def parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def iso(d: datetime) -> str:
    return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def load_state(path: Path = STATE) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def next_reset(now: datetime, day: int) -> datetime:
    """The next time the monthly quota resets (00:00 UTC on `day` of the month)."""
    y, m, day = now.year, now.month, max(1, min(int(day), 28))
    while True:
        cand = datetime(y, m, day, tzinfo=timezone.utc)
        if cand > now:
            return cand
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def milestone_key(k: datetime, minutes: int) -> str:
    return f"{iso(k)}|{minutes}"


def due_milestones(state: dict, now: datetime, cfg: dict) -> list[str]:
    s = settings(cfg)
    done = set(state.get("milestones_done", []))
    out = []
    for ks in state.get("kickoffs", []):
        k = parse(ks)
        for m in s["milestones_min"]:
            if k - timedelta(minutes=m) <= now < k - timedelta(minutes=2) and milestone_key(k, m) not in done:
                out.append(milestone_key(k, m))
    return out


def plan(state: dict, now: datetime, cfg: dict) -> dict:
    """Budget figures used by decide(); also shown on the site."""
    s = settings(cfg)
    cost_full = len(cfg.get("odds_api_regions", ["uk", "eu"]))
    cost_focus = 1
    remaining = state.get("remaining")
    reset = next_reset(now, int(state.get("reset_day") or cfg.get("odds_api_reset_day", 1)))
    hours_left = max(1.0, (reset - now).total_seconds() / 3600)
    days_left = hours_left / 24
    upcoming = sorted(k for k in (parse(x) for x in state.get("kickoffs", [])) if k and k > now)
    planned_milestones = math.ceil(days_left * s["slots_per_day"]) * len(s["milestones_min"]) * cost_focus
    planned_full = math.ceil(hours_left / s["full_every_h"]) * cost_full
    out = {"remaining": remaining, "reset": iso(reset), "hours_left": round(hours_left, 1),
           "planned_milestones": planned_milestones, "planned_full": planned_full,
           "cost_full": cost_full, "cost_focus": cost_focus,
           "next_kickoff": iso(upcoming[0]) if upcoming else None}
    if remaining is None:
        out.update(spare=None, gap_h=None)
        return out
    spare = remaining - planned_full - planned_milestones - s["reserve"]
    active_left = hours_left * s["active_hours"] / 24
    gap = s["max_gap_h"] if spare <= 0 else min(s["max_gap_h"], max(s["min_gap_h"], active_left / spare))
    if not upcoming or upcoming[0] - now > timedelta(days=5):
        gap = max(gap, s["quiet_gap_h"])
    out.update(spare=spare, gap_h=round(gap, 2))
    return out


def decide(state: dict, now: datetime, cfg: dict, event: str = "schedule", force: str = "auto") -> tuple[str, str]:
    """Return (mode, reason). mode is full, focus, build (no credits) or skip."""
    if force in ("full", "focus", "none"):
        return ("build" if force == "none" else force), f"forced by the person running the workflow ({force})"
    s = settings(cfg)
    fetches = state.get("fetches", [])
    last_any = parse(fetches[-1]["at"]) if fetches else None
    last_full = next((parse(f["at"]) for f in reversed(fetches) if f.get("mode") == "full"), None)
    last_build = parse(state.get("last_build"))
    p = plan(state, now, cfg)
    rem = p["remaining"]

    def fallback(reason: str) -> tuple[str, str]:
        if event != "schedule":
            return "build", f"{event}: rebuild without new odds ({reason})"
        if last_build is None or now - last_build >= timedelta(hours=s["rebuild_every_h"]):
            return "build", f"site older than {s['rebuild_every_h']}h ({reason})"
        return "skip", reason

    if rem is None or last_full is None:
        return "full", "no snapshot yet"
    if rem < p["cost_focus"] + 1:
        return fallback(f"only {rem} credits left until {p['reset']}")
    due = due_milestones(state, now, cfg)
    if due:
        return "focus", "kick-off snapshot " + ", ".join(due)
    if now - last_full >= timedelta(hours=s["full_every_h"]) and rem - p["cost_full"] >= p["planned_milestones"] + s["reserve"]:
        return "full", f"last full snapshot {(now - last_full).total_seconds() / 3600:.1f}h ago"
    if p["spare"] > 0 and now - last_any >= timedelta(hours=p["gap_h"]):
        return "focus", f"every {p['gap_h']}h to spread {p['spare']} spare credits until {p['reset']}"
    return fallback(f"last snapshot {(now - last_any).total_seconds() / 60:.0f} min ago, next in {p['gap_h']}h")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="schedule")
    ap.add_argument("--force", default="auto")
    ap.add_argument("--now")
    a = ap.parse_args()
    cfg = json.loads((ROOT / "config.json").read_text())
    now = parse(a.now) if a.now else datetime.now(timezone.utc)
    mode, reason = decide(load_state(), now, cfg, a.event, a.force or "auto")
    print(f"mode={mode}")
    print(f"reason={reason}")


if __name__ == "__main__":
    main()
