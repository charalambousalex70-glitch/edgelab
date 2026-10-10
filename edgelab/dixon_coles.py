"""Dixon & Coles (1997) match model.

Home goals X ~ Poisson(lambda), away goals Y ~ Poisson(mu), with
    lambda = alpha_home * beta_away * gamma
    mu     = alpha_away * beta_home
and the low-score correction tau(x, y) on 0-0, 1-0, 0-1 and 1-1 with
dependence parameter rho (paper eq. 4.2). Past matches are down-weighted by
exp(-xi * age) with age in half-weeks (eq. 4.5). Identification: mean(alpha) = 1.

Fitted by maximising the weighted log-likelihood with an analytic gradient.
Parameters are optimised on the log scale (alpha, beta, gamma > 0 automatically).

Two additions that are ours, not the paper's, both off by default:
  * `shrink`: an L2 penalty pulling log-ratings towards the league average.
    Useful early in a season or for promoted teams with little history.
  * `response="xg"`: fit the attack/defence/home parameters to xG instead of
    goals (Poisson quasi-likelihood). tau is undefined for non-integer scores,
    so rho is then held fixed (estimate it from goals first).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

HALF_WEEK_DAYS = 3.5
MAX_GOALS = 10
PAPER_XI = 0.0065  # per half-week, Dixon & Coles section 4.4


@dataclass
class DixonColesFit:
    teams: list[str]
    attack: dict[str, float]
    defence: dict[str, float]
    home: float
    rho: float
    xi: float
    as_of: pd.Timestamp
    n_matches: int
    log_likelihood: float
    converged: bool
    _index: dict[str, int] = field(default_factory=dict, repr=False)
    priors: dict[str, tuple[float, float]] = field(default_factory=dict, repr=False)

    # ---------- prediction ----------
    def _rating(self, team: str) -> tuple[float, float]:
        if team in self.attack:
            return self.attack[team], self.defence[team]
        if team in self.priors:  # e.g. a promoted side before its first match
            la, lb = self.priors[team]
            return float(np.exp(la)), float(np.exp(lb))
        raise KeyError(f"No rating for {team!r}; it has no matches in the fitting window and no prior")

    def rates(self, home_team: str, away_team: str) -> tuple[float, float]:
        ah, dh = self._rating(home_team)
        aa, da = self._rating(away_team)
        return ah * da * self.home, aa * dh

    def score_matrix(self, home_team: str, away_team: str, max_goals: int = MAX_GOALS) -> np.ndarray:
        lam, mu = self.rates(home_team, away_team)
        return score_matrix(lam, mu, self.rho, max_goals)

    def predict(self, home_team: str, away_team: str) -> dict[str, float]:
        lam, mu = self.rates(home_team, away_team)
        m = score_matrix(lam, mu, self.rho)
        return {"home_xg": lam, "away_xg": mu, **markets(m)}

    def ratings_table(self) -> pd.DataFrame:
        df = pd.DataFrame({"team": self.teams,
                           "attack": [self.attack[t] for t in self.teams],
                           "defence": [self.defence[t] for t in self.teams]})
        # Net strength on a neutral pitch vs an average side: log attack - log defence.
        df["net"] = np.log(df.attack) - np.log(df.defence)
        return df.sort_values("net", ascending=False).reset_index(drop=True)


def tau_matrix(lam: float, mu: float, rho: float, max_goals: int) -> np.ndarray:
    t = np.ones((max_goals + 1, max_goals + 1))
    t[0, 0] = 1 - lam * mu * rho
    t[0, 1] = 1 + lam * rho
    t[1, 0] = 1 + mu * rho
    t[1, 1] = 1 - rho
    return t


def score_matrix(lam: float, mu: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """P(home = i, away = j) for i, j in 0..max_goals, renormalised for truncation."""
    g = np.arange(max_goals + 1)
    ph = np.exp(-lam + g * np.log(lam) - gammaln(g + 1))
    pa = np.exp(-mu + g * np.log(mu) - gammaln(g + 1))
    m = np.outer(ph, pa) * tau_matrix(lam, mu, rho, max_goals)
    m = np.clip(m, 0, None)
    return m / m.sum()


def markets(m: np.ndarray) -> dict[str, float]:
    i, j = np.indices(m.shape)
    return {
        "home": float(m[i > j].sum()),
        "draw": float(m[i == j].sum()),
        "away": float(m[i < j].sum()),
        "over_2_5": float(m[i + j >= 3].sum()),
        "btts": float(m[(i > 0) & (j > 0)].sum()),
    }


def rho_bounds(lam: np.ndarray, mu: np.ndarray) -> tuple[float, float]:
    """Paper's constraint: max(-1/lam, -1/mu) <= rho <= min(1/(lam*mu), 1), over all matches."""
    lo = float(np.max(np.maximum(-1 / lam, -1 / mu)))
    hi = float(min(np.min(1 / (lam * mu)), 1.0))
    return lo, hi


def _prepare(matches: pd.DataFrame, as_of, xi: float, response: str):
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"])
    as_of = pd.Timestamp(as_of) if as_of is not None else df["date"].max() + pd.Timedelta(days=1)
    df = df[df["date"] < as_of]
    if df.empty:
        raise ValueError("No matches before as_of")
    cols = ("home_goals", "away_goals") if response == "goals" else ("home_xg", "away_xg")
    df = df.dropna(subset=list(cols))
    teams = sorted(set(df["home"]) | set(df["away"]))
    idx = {t: k for k, t in enumerate(teams)}
    h = df["home"].map(idx).to_numpy()
    a = df["away"].map(idx).to_numpy()
    x = df[cols[0]].to_numpy(float)
    y = df[cols[1]].to_numpy(float)
    age = (as_of - df["date"]).dt.total_seconds().to_numpy() / 86400 / HALF_WEEK_DAYS
    w = np.exp(-xi * age)
    return teams, idx, h, a, x, y, w, as_of


def _nll_and_grad(theta, n, h, a, x, y, w, fit_rho, fixed_rho, shrink, use_tau, pm=None, pw=None):
    la, lb = theta[:n], theta[n:2 * n]
    lg = theta[2 * n]
    rho = theta[2 * n + 1] if fit_rho else fixed_rho
    log_lam = la[h] + lb[a] + lg
    log_mu = la[a] + lb[h]
    lam, mu = np.exp(log_lam), np.exp(log_mu)

    # Poisson part (constants dropped; fine for xG too as a quasi-likelihood).
    ll = w * (x * log_lam - lam + y * log_mu - mu)
    d_loglam = w * (x - lam)
    d_logmu = w * (y - mu)
    d_rho = 0.0

    if use_tau:
        m00 = (x == 0) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0)
        m11 = (x == 1) & (y == 1)
        tau = np.ones_like(lam)
        tau[m00] = 1 - lam[m00] * mu[m00] * rho
        tau[m01] = 1 + lam[m01] * rho
        tau[m10] = 1 + mu[m10] * rho
        tau[m11] = 1 - rho
        if np.any(tau <= 0):
            return 1e12, np.zeros_like(theta)
        ll = ll + w * np.log(tau)
        g = w / tau
        d_loglam[m00] += g[m00] * (-lam[m00] * mu[m00] * rho)
        d_logmu[m00] += g[m00] * (-lam[m00] * mu[m00] * rho)
        d_loglam[m01] += g[m01] * (lam[m01] * rho)
        d_logmu[m10] += g[m10] * (mu[m10] * rho)
        d_rho = (np.sum(g[m00] * (-lam[m00] * mu[m00])) + np.sum(g[m01] * lam[m01])
                 + np.sum(g[m10] * mu[m10]) - np.sum(g[m11]))

    grad = np.zeros_like(theta)
    np.add.at(grad, h, d_loglam)           # attack of home side
    np.add.at(grad, n + a, d_loglam)       # defence of away side
    grad[2 * n] = d_loglam.sum()           # home effect
    np.add.at(grad, a, d_logmu)            # attack of away side
    np.add.at(grad, n + h, d_logmu)        # defence of home side
    if fit_rho:
        grad[2 * n + 1] = d_rho

    total = ll.sum()
    if shrink > 0:
        total -= shrink * (np.sum(la ** 2) + np.sum(lb ** 2))
        grad[:2 * n] -= 2 * shrink * theta[:2 * n]
    if pm is not None:  # per-team prior: pw * ((la - m_a)^2 + (lb - m_b)^2)
        da, db = la - pm[:, 0], lb - pm[:, 1]
        total -= np.sum(pw * (da ** 2 + db ** 2))
        grad[:n] -= 2 * pw * da
        grad[n:2 * n] -= 2 * pw * db
    # Soft identification: keep mean log-attack near 0 (exact rescaling happens after).
    total -= 1e3 * la.mean() ** 2
    grad[:n] -= 1e3 * 2 * la.mean() / n
    return -total, -grad


def fit(matches: pd.DataFrame, *, as_of=None, xi: float = PAPER_XI, response: str = "goals",
        rho: float | None = None, shrink: float = 0.0, init: DixonColesFit | None = None,
        priors: dict[str, tuple[float, float]] | None = None, prior_weight: float = 5.0) -> DixonColesFit:
    """Fit Dixon-Coles using matches strictly before `as_of`.

    `matches` needs columns: date, home, away, and home_goals/away_goals
    (or home_xg/away_xg when response="xg").

    `priors` maps team -> (log attack, log defence) prior means, applied with
    `prior_weight` (about 4 per match of information it is worth; 5 is roughly
    6-8 matches). Ours, not the paper's: used for promoted sides with no history.
    """
    if response not in ("goals", "xg"):
        raise ValueError("response must be 'goals' or 'xg'")
    teams, idx, h, a, x, y, w, as_of = _prepare(matches, as_of, xi, response)
    n = len(teams)
    use_tau = response == "goals"
    fit_rho = use_tau and rho is None
    fixed_rho = 0.0 if rho is None else rho

    theta0 = np.zeros(2 * n + 2)
    theta0[2 * n] = np.log(1.3)
    theta0[2 * n + 1] = -0.05
    if init is not None:  # warm start from a previous fit
        for t, k in idx.items():
            if t in init.attack:
                theta0[k] = np.log(init.attack[t])
                theta0[n + k] = np.log(init.defence[t])
        theta0[2 * n], theta0[2 * n + 1] = np.log(init.home), init.rho

    pm = pw = None
    if priors:
        pm = np.zeros((n, 2)); pw = np.zeros(n)
        for t, k in idx.items():
            if t in priors:
                pm[k] = priors[t]; pw[k] = prior_weight
                if init is None or t not in init.attack:
                    theta0[k], theta0[n + k] = priors[t]
    bounds = [(None, None)] * (2 * n + 1) + [(-0.5, 0.5)]
    res = minimize(_nll_and_grad, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                   args=(n, h, a, x, y, w, fit_rho, fixed_rho, shrink, use_tau, pm, pw),
                   options={"maxiter": 2000})

    alpha = np.exp(res.x[:n])
    beta = np.exp(res.x[n:2 * n])
    c = alpha.mean()  # impose the paper's constraint mean(alpha) = 1 exactly
    alpha, beta = alpha / c, beta * c
    rho_hat = float(res.x[2 * n + 1]) if fit_rho else fixed_rho
    return DixonColesFit(
        teams=teams,
        attack=dict(zip(teams, alpha.tolist())),
        defence=dict(zip(teams, beta.tolist())),
        home=float(np.exp(res.x[2 * n])),
        rho=rho_hat, xi=xi, as_of=as_of, n_matches=len(h),
        log_likelihood=float(-res.fun), converged=bool(res.success), _index=idx,
        priors=dict(priors or {}),
    )


def rolling_predictions(matches: pd.DataFrame, *, xi: float = PAPER_XI, min_history_days: float = 60 * HALF_WEEK_DAYS,
                        response: str = "goals", rho: float | None = None, shrink: float = 0.0,
                        refit_every_days: int = 1, priors: dict | None = None,
                        prior_weight: float = 5.0, start=None) -> pd.DataFrame:
    """Out-of-sample predictions: each matchday is predicted from a fit on earlier matches only.

    Mirrors the paper's pseudo real-time procedure (no hindsight). Teams with no
    history in the window (e.g. newly promoted with no data) are skipped and flagged.
    """
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    start = pd.Timestamp(start) if start is not None else df["date"].min() + pd.Timedelta(days=min_history_days)
    rows, last_fit, last_fit_day = [], None, None
    for day, games in df[df["date"] >= start].groupby(df["date"].dt.normalize()):
        if last_fit is None or (day - last_fit_day).days >= refit_every_days:
            last_fit = fit(df, as_of=day, xi=xi, response=response, rho=rho, shrink=shrink, init=last_fit,
                           priors=priors, prior_weight=prior_weight)
            last_fit_day = day
        for _, g in games.iterrows():
            # Team names go in home_team/away_team; "home"/"draw"/"away" hold probabilities.
            rec = {"date": g["date"], "home_team": g["home"], "away_team": g["away"],
                   "home_goals": g.get("home_goals"), "away_goals": g.get("away_goals")}
            try:
                rec.update(last_fit.predict(g["home"], g["away"]))
            except KeyError:
                rec["skipped"] = "no history"
            rows.append(rec)
    return pd.DataFrame(rows)


def outcome_log_score(preds: pd.DataFrame) -> float:
    """Paper's S(xi), eq. 4.7: sum of log-probabilities of the observed outcomes."""
    p = preds.dropna(subset=["home", "home_goals"])
    hg, ag = p["home_goals"].to_numpy(), p["away_goals"].to_numpy()
    prob = np.where(hg > ag, p["home"], np.where(hg == ag, p["draw"], p["away"]))
    return float(np.log(np.clip(prob, 1e-12, None)).sum())


def rps(preds: pd.DataFrame) -> float:
    """Mean ranked probability score for home/draw/away (lower is better)."""
    p = preds.dropna(subset=["home", "home_goals"])
    hg, ag = p["home_goals"].to_numpy(), p["away_goals"].to_numpy()
    o = np.stack([(hg > ag), (hg == ag), (hg < ag)], axis=1).astype(float)
    f = p[["home", "draw", "away"]].to_numpy()
    cf, co = np.cumsum(f, axis=1)[:, :2], np.cumsum(o, axis=1)[:, :2]
    return float(np.mean(np.sum((cf - co) ** 2, axis=1) / 2))
