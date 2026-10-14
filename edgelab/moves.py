"""Price movements and slow bookmakers, from the saved Odds API snapshots.

Definitions (all probabilities have the margin removed proportionally):
  sharp price  the average of Pinnacle and the Betfair Exchange at a snapshot.
  sharp move   between two snapshots in a row, the sharp chance of some result changes
               by at least `move_min` (1.5 percentage points).
  follow       how much of a sharp move a bookmaker had copied by the same snapshot:
               its own change in that result's chance divided by the sharp change
               (100% = fully copied, 0% = not moved).
  catch-up     hours from the sharp move (dated by Pinnacle/exchange `last_update`) until
               the bookmaker had copied at least half of it (dated by its own
               `last_update`). Moves the bookmaker never copied within 24 hours, or before
               kick-off, count as "never" in the median.
  closing price  the last sharp price taken within 3 hours of kick-off.
  CLV          closing line value of a price: closing chance x odds - 1. Positive means the
               price beat the market's final view.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .snapshots import EXCHANGE_KEYS, SHARP_KEYS

P = ["p0", "p1", "p2"]
BUCKETS = [(72, 1e9, "Over 3 days"), (24, 72, "1 to 3 days"), (6, 24, "6 to 24 hours"), (0, 6, "Last 6 hours")]
SEGMENTS = [(None, 72, "Opening to 3 days before"), (72, 24, "3 days to 1 day before"),
            (24, 6, "Last day, to 6 hours before"), (6, 1, "6 hours to 1 hour before"),
            (1, 0, "Last hour (line-ups)")]


def r4(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def prepare(s: pd.DataFrame) -> pd.DataFrame:
    """Pre-kick-off rows with margin-free chances."""
    s = s[s.fetched < s.commence - timedelta(minutes=2)].copy()
    if s.empty:
        return s
    inv = 1 / s[["h", "d", "a"]].to_numpy(float)
    tot = inv.sum(1, keepdims=True)
    s[P] = inv / tot
    s["margin"] = tot[:, 0] - 1
    s["match"] = s.home + " v " + s.away + " " + s.commence.dt.strftime("%Y-%m-%d")
    s["sharp"] = s.book_key.isin(SHARP_KEYS)
    s["exch"] = s.book_key.isin(EXCHANGE_KEYS)
    return s


def sharp_series(s: pd.DataFrame) -> pd.DataFrame:
    """One row per match and snapshot: the sharp chances and when the sharp price last changed."""
    sh = s[s.sharp].copy()
    if sh.empty:
        return pd.DataFrame(columns=["match", "fetched", "home", "away", "commence", "lu", *P])
    sh["src"] = np.where(sh.book_key == "pinnacle", "pin", "bfe")
    sh = sh.sort_values("book_key", ascending=False).drop_duplicates(["match", "fetched", "src"])  # exchange uk before eu
    g = sh.groupby(["match", "fetched"], as_index=False).agg(
        home=("home", "first"), away=("away", "first"), commence=("commence", "first"), lu=("last_update", "max"),
        p0=("p0", "mean"), p1=("p1", "mean"), p2=("p2", "mean"), n_src=("src", "nunique"))
    return g.sort_values(["match", "fetched"]).reset_index(drop=True)


def _soft(s: pd.DataFrame) -> pd.DataFrame:
    return s[~s.sharp & ~s.exch]


def lag_events(s: pd.DataFrame, sp: pd.DataFrame, latest: pd.Timestamp, move_min: float = 0.015,
               window_h: float = 24) -> pd.DataFrame:
    """One row per (bookmaker, two snapshots in a row of that bookmaker, sharp move between
    them). Focus bookmakers are seen every couple of hours, the rest only in full snapshots
    (every 12 hours), so `gap_h` records how far apart the two snapshots were."""
    soft = _soft(s)
    soft_by = {m: x for m, x in soft.groupby("match")}
    win = np.timedelta64(int(window_h * 3600), "s")
    hour = np.timedelta64(3600, "s")
    latest = np.datetime64(latest)
    out = []
    for match, g in sp.groupby("match"):
        if len(g) < 2 or match not in soft_by:
            continue
        g = g.sort_values("fetched")
        gt, GQ, GLU = g.fetched.to_numpy(), g[P].to_numpy(float), g.lu.to_numpy()
        pos = {t: i for i, t in enumerate(gt)}
        kick = g.commence.to_numpy()[0]
        for book, b in soft_by[match].groupby("book"):
            b = b[b.fetched.isin(gt)].sort_values("fetched")
            if len(b) < 2:
                continue
            bt, BP, BO, BLU = b.fetched.to_numpy(), b[P].to_numpy(float), b[["h", "d", "a"]].to_numpy(float), b.last_update.to_numpy()
            gi = np.array([pos[t] for t in bt])
            dQ = GQ[gi[1:]] - GQ[gi[:-1]]
            ks = np.abs(dQ).argmax(1)
            for i in np.where(np.abs(dQ[np.arange(len(ks)), ks]) >= move_min)[0] + 1:
                k = int(ks[i - 1]); dq = dQ[i - 1, k]
                t0, t1 = bt[i - 1], bt[i]
                lu = GLU[gi[i]]
                t_sharp = lu if not np.isnat(lu) and t0 < lu <= t1 else t1
                end = min(t0 + win, kick)
                after = np.where((gt > t1) & (gt <= end))[0]
                hold = GQ[after[-1], k] if len(after) else GQ[gi[i], k]
                confirmed = (hold - GQ[gi[i - 1], k]) / dq >= 0.5  # the move had not reversed by the end
                base = BP[i - 1, k]
                lag, status, prev_t = np.nan, "pending", t0
                for j in range(i, len(bt)):
                    if bt[j] > end:
                        break
                    if (BP[j, k] - base) / dq >= 0.5:
                        blu = BLU[j]
                        t_book = blu if not np.isnat(blu) and prev_t < blu <= bt[j] else bt[j]
                        lag, status = (t_book - t_sharp) / hour, "followed"
                        break
                    prev_t = bt[j]
                if status == "pending" and (latest >= end or bt[-1] >= end - hour / 2):
                    status = "never"
                out.append({"match": match, "book": book, "t1": pd.Timestamp(t1), "k": k, "dq": dq,
                            "gap_h": (t1 - t0) / hour,
                            "follow": float(np.clip((BP[i, k] - base) / dq, -0.5, 1.5)),
                            "lag_h": lag, "status": status, "confirmed": bool(confirmed),
                            "ev_after": GQ[gi[i], k] * BO[i, k] - 1})
    return pd.DataFrame(out)


def _mean_ci(x, cluster=None):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x = x[ok]
    if len(x) < 2:
        return (r4(x.mean()) if len(x) else None), None
    if cluster is not None:  # standard error from per-match means (prices within a match move together)
        cl = np.asarray(cluster)[ok]
        means = pd.Series(x).groupby(cl).mean()
        se = means.std(ddof=1) / np.sqrt(len(means)) if len(means) > 1 else np.nan
    else:
        se = x.std(ddof=1) / np.sqrt(len(x))
    return r4(x.mean()), r4(1.645 * se) if np.isfinite(se) else None


def closing(sp: pd.DataFrame, max_h: float = 3) -> pd.DataFrame:
    """The last sharp price before kick-off, when it was taken within `max_h` hours of it."""
    last = sp.sort_values("fetched").groupby("match").tail(1)
    last = last[(last.commence - last.fetched) <= timedelta(hours=max_h)]
    return last.set_index("match")[P]


def with_sharp(s: pd.DataFrame, sp: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Bookmaker rows with the sharp chances of the same snapshot and their value then."""
    soft = _soft(s).merge(sp[["match", "fetched", *P]].rename(columns={p: "q" + p[1] for p in P}), on=["match", "fetched"])
    odds = soft[["h", "d", "a"]].to_numpy(float)
    q = soft[["q0", "q1", "q2"]].to_numpy(float)
    ev = q * odds - 1
    soft[["ev0", "ev1", "ev2"]] = ev
    soft["value"] = (ev >= threshold).any(1)
    return soft


def book_table(s, sp, ev: pd.DataFrame, ws: pd.DataFrame, clv_rows: pd.DataFrame, threshold: float) -> list:
    rows = []
    for book, g in ws.groupby("book"):
        e = ev[(ev.book == book) & ev.confirmed] if len(ev) else ev
        n_obs = len(g) * 3
        val = g[["ev0", "ev1", "ev2"]].to_numpy()
        f, f_ci = _mean_ci(e.follow, e.match) if len(e) else (None, None)
        lags = e[e.status != "pending"]
        if len(lags):
            lag_vals = np.where(lags.status == "followed", lags.lag_h.clip(lower=-6), np.inf)
            med = float(np.median(lag_vals))
            never = float((lags.status == "never").mean())
        else:
            med, never = np.nan, None
        c = clv_rows[clv_rows.book == book]
        clv, clv_ci = _mean_ci(c.clv, c.match) if len(c) else (None, None)
        rows.append({"name": book, "snapshots": int(g.fetched.nunique()), "matches": int(g.match.nunique()),
                     "margin": r4(g.margin.mean()), "moves": int(len(e)), "follow": f, "follow_ci": f_ci,
                     "gap_h": r4(float(e.gap_h.median())) if len(e) else None,
                     "catchup_h": None if not np.isfinite(med) else round(med, 2), "never": r4(never) if never is not None else None,
                     "never_inf": bool(len(lags)) and not np.isfinite(med),
                     "value_rate": r4((val >= threshold).sum() / n_obs) if n_obs else None,
                     "value_after_move": r4((e.ev_after >= threshold).mean()) if len(e) else None,
                     "clv": clv, "clv_ci": clv_ci, "clv_n": int(c.match.nunique()) if len(c) else 0})
    return sorted(rows, key=lambda r: (r["follow"] is None, (r["gap_h"] or 0) > 4, r["follow"] if r["follow"] is not None else 0))


def clv_rows(ws: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """Each bookmaker price (averaged over the three results) scored against the closing sharp price."""
    c = ws[ws.match.isin(close.index)].copy()
    if c.empty:
        return c.assign(clv=[], hours=[])
    pc = close.loc[c.match].to_numpy()
    odds = c[["h", "d", "a"]].to_numpy(float)
    c[["c0", "c1", "c2"]] = pc * odds - 1
    c["clv"] = c[["c0", "c1", "c2"]].mean(1)
    c["hours"] = (c.commence - c.fetched).dt.total_seconds() / 3600
    return c


def timing(sp: pd.DataFrame, close: pd.DataFrame) -> list:
    """How far the sharp price moved in each period before kick-off (half the sum of the
    absolute changes in the three chances), averaged over finished matches."""
    out = {lab: [] for *_, lab in SEGMENTS}
    for match, g in sp[sp.match.isin(close.index)].groupby("match"):
        kick = g.commence.iloc[0]

        def at(h):
            if h is None:
                return g[P].iloc[0].to_numpy()
            x = g[g.fetched <= kick - timedelta(hours=h)]
            return x[P].iloc[-1].to_numpy() if len(x) else None
        for start, stop, lab in SEGMENTS:
            a, b = at(start), (close.loc[match].to_numpy() if stop == 0 else at(stop))
            first = g.fetched.iloc[0]
            if a is None or b is None or (start is not None and first > kick - timedelta(hours=start)):
                continue
            out[lab].append(0.5 * np.abs(b - a).sum())
    return [{"label": lab, "n": len(v), "move": r4(np.mean(v)) if v else None} for lab, v in out.items()]


def clv_by_time(c: pd.DataFrame, close: pd.DataFrame, threshold: float) -> list:
    """By time before kick-off: CLV of the best bookmaker price, and of value bets flagged then."""
    out = []
    if c.empty:
        return [{"label": lab, "n": 0} for *_, lab in BUCKETS]
    for lo, hi, lab in BUCKETS:
        b = c[(c.hours >= lo) & (c.hours < hi)]
        row = {"label": lab, "n": int(b.match.nunique())}
        if len(b):
            best = b.groupby(["match", "fetched"])[["h", "d", "a"]].max()
            m = best.index.get_level_values(0)
            bclv = (close.loc[m].to_numpy() * best.to_numpy(float) - 1).mean(1)
            row["best"], row["best_ci"] = _mean_ci(bclv, m)
        flag = b[["ev0", "ev1", "ev2"]].to_numpy() >= threshold
        cv = b[["c0", "c1", "c2"]].to_numpy()[flag]
        mm = np.repeat(b.match.to_numpy()[:, None], 3, 1)[flag]
        row["value_n"] = int(flag.sum())
        row["value_clv"], row["value_ci"] = _mean_ci(cv, mm) if flag.any() else (None, None)
        out.append(row)
    return out


def match_cards(s: pd.DataFrame, sp: pd.DataFrame, now: pd.Timestamp, threshold: float, move_min: float = 0.015,
                max_stale_h: float = 3) -> list:
    """For each upcoming match: the sharp price over time, how far it has moved, and which
    bookmakers have not followed the latest move (with the value that leaves)."""
    soft = _soft(s)
    cards = []
    for match, g in sp[sp.commence > now].groupby("match"):
        g = g.sort_values("fetched")
        Q = g[P].to_numpy()
        ref = g[g.fetched <= g.fetched.iloc[-1] - timedelta(hours=12)]
        q_ref = (ref[P].iloc[-1] if len(ref) else g[P].iloc[0]).to_numpy()
        t_ref = (ref.fetched.iloc[-1] if len(ref) else g.fetched.iloc[0])
        d12 = Q[-1] - q_ref
        k = int(np.argmax(np.abs(d12)))
        lagging = []
        if abs(d12[k]) >= move_min:
            t_now = g.fetched.iloc[-1]
            mine = soft[soft.match == match]
            for book, b in mine.groupby("book"):
                b = b.sort_values("fetched")
                cur, old = b[b.fetched >= t_now - timedelta(hours=max_stale_h)], b[b.fetched <= t_ref]
                if cur.empty or old.empty:
                    continue
                cur, old = cur.iloc[-1], old.iloc[-1]
                moved = (cur[P[k]] - old[P[k]]) / d12[k]
                gap = (Q[-1][k] - cur[P[k]]) * np.sign(d12[k])  # still behind the sharp price, in the move's direction
                price = float(cur[["h", "d", "a"][k]])
                if moved < 0.7 and gap >= 0.01:
                    lagging.append({"book": book, "moved": r4(moved), "gap": r4(gap), "price": price,
                                    "was": float(old[["h", "d", "a"][k]]), "ev": r4(Q[-1][k] * price - 1),
                                    "seen": cur.fetched.strftime("%Y-%m-%dT%H:%MZ")})
            lagging.sort(key=lambda x: -x["ev"])
        fx = g.fetched.dt.tz_localize("UTC").dt.tz_convert("Europe/London")
        step = max(1, len(g) // 80)
        cards.append({"home": g.home.iloc[0], "away": g.away.iloc[0],
                      "date": g.commence.iloc[0].tz_localize("UTC").tz_convert("Europe/London").strftime("%Y-%m-%dT%H:%M"),
                      "series": [[t.strftime("%Y-%m-%dT%H:%M"), *[round(float(v), 4) for v in q]]
                                 for t, q in list(zip(fx, Q))[::step] + ([(fx.iloc[-1], Q[-1])] if (len(g) - 1) % step else [])],
                      "open": [r4(v) for v in Q[0]], "now": [r4(v) for v in Q[-1]],
                      "d12": [r4(v) for v in d12], "k12": k, "lagging": lagging[:8], "n": int(len(g))})
    cards.sort(key=lambda c: -max(abs(c["now"][i] - c["open"][i]) for i in range(3)))
    return cards


def analyse(snap: pd.DataFrame, now: pd.Timestamp, threshold: float, move_min: float = 0.015) -> dict:
    s = prepare(snap)
    out = {"fetches": int(snap.fetched.nunique()) if len(snap) else 0,
           "first": None if snap.empty else snap.fetched.min().strftime("%Y-%m-%dT%H:%MZ"),
           "last": None if snap.empty else snap.fetched.max().strftime("%Y-%m-%dT%H:%MZ"),
           "move_min": move_min, "matches": [], "books": [], "timing": [], "clv_time": [], "events": 0, "closed": 0}
    if s.empty:
        return out
    sp = sharp_series(s)
    if sp.empty:
        return out
    ev = lag_events(s, sp, snap.fetched.max(), move_min)
    ws = with_sharp(s, sp, threshold)
    close = closing(sp)
    c = clv_rows(ws, close)
    out.update(matches=match_cards(s, sp, now, threshold, move_min),
               books=book_table(s, sp, ev, ws, c, threshold), timing=timing(sp, close),
               clv_time=clv_by_time(c, close, threshold),
               events=int(ev[["match", "t1"]].drop_duplicates().shape[0]) if len(ev) else 0, closed=int(len(close)))
    return out


def context(cards: list, home: str, away: str, k: int, book_names: list[str]) -> str | None:
    """One line for a value bet: has the sharp price just moved, and has this bookmaker followed?"""
    c = next((x for x in cards if x["home"] == home and x["away"] == away), None)
    if not c or c["n"] < 2:
        return None
    d = c["d12"][k]
    lag = [x for x in c["lagging"] if x["book"] in book_names]
    if abs(d) < 0.015:
        return "The exchange price has been steady over the last 12 hours, so this is a soft price rather than a slow one."
    way = "shortened" if d > 0 else "drifted"
    txt = f"The sharp price for this result has {way} by {abs(d) * 100:.1f} points in the last 12 hours"
    if lag:
        x = lag[0]
        how = (f"has only partly followed ({x['was']:.2f} to {x['price']:.2f})" if x["moved"] > 0.15
               else f"has not followed (still {x['price']:.2f})")
        return txt + f"; {x['book']} {how}. Slow prices like this tend to be cut soon."
    return txt + "."
