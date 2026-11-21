"""Film scores that allow for how many ratings a film has.

The model is the same one used for surgical case durations in the theatre scheduler project:

    film i's average rating:   ybar_i ~ Normal(theta_i, sigma^2 / n_i)     (n_i ratings, each with spread sigma)
    film i's true score:       theta_i ~ Normal(x_i' gamma, tau^2)         (x_i: genre and decade, or just 1)

theta_i is the average the film would settle at with unlimited ratings. tau is how far true scores spread
around what the film's genre and decade predict. sigma is how far individual ratings of one film spread.

Given tau, each film's posterior mean is a weighted average of its own average and the prediction:

    w_i = n_i / (n_i + sigma^2 / tau^2),     posterior mean = w_i * ybar_i + (1 - w_i) * x_i' gamma

so sigma^2 / tau^2 behaves like a number of imaginary ratings at the predicted value. tau and gamma are
estimated from all films together: gamma has a flat prior, tau a uniform prior on a grid, and the posterior
of tau on the grid follows Gelman et al., Bayesian Data Analysis (3rd edn), section 5.4.

The IMDb-style formula, n / (n + m) * ybar + m / (n + m) * C, is the special case with no predictors and a
fixed m in place of the estimated sigma^2 / tau^2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import ndtri

DECADES = list(range(1920, 2030, 10))


def raw_score(summary: pd.DataFrame) -> pd.Series:
    """The plain average rating: what most sites show."""
    return summary["mean"].rename("raw")


def imdb_score(summary: pd.DataFrame, m: float, overall: float) -> pd.Series:
    """The IMDb-style weighted rating: n / (n + m) * film average + m / (n + m) * overall average."""
    if m < 0:
        raise ValueError("m must be zero or positive")
    n = summary["n"].to_numpy(dtype=float)
    return pd.Series((n * summary["mean"].to_numpy() + m * overall) / (n + m), index=summary.index, name="imdb")


def pooled_sigma2(summary: pd.DataFrame) -> float:
    """Spread of individual ratings within a film, pooled over films with two or more ratings."""
    ok = summary["n"] > 1
    dof = (summary.loc[ok, "n"] - 1).to_numpy(dtype=float)
    if dof.sum() == 0:
        raise ValueError("need at least one film with two or more ratings to estimate sigma")
    return float((dof * summary.loc[ok, "var"].to_numpy()).sum() / dof.sum())


COUNT_BANDS = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 25), (26, 100), (101, 1000)]  # above 1,000: reference


def design(films: pd.DataFrame, index: pd.Index, kind: str = "genre", counts: pd.Series | None = None
           ) -> pd.DataFrame:
    """Predictors for each film's prior mean, one row per film in index.

    kind "flat": an intercept only, so every film is pulled towards the same overall average.
    kind "genre": an intercept, one indicator per genre (films can have several) and one per release decade.
    kind "catalogue": as "genre", plus bands of the film's number of ratings (counts, aligned to index).
        Films few people rate tend to be rated lower, so for a whole catalogue the count is evidence too.
        Not for the thin-data tests, where a popular film is deliberately shown only a few ratings.
    """
    if kind == "flat":
        return pd.DataFrame({"intercept": 1.0}, index=index)
    if kind == "catalogue":
        if counts is None:
            raise ValueError("kind 'catalogue' needs counts")
        X = design(films, index, "genre")
        c = counts.reindex(index).to_numpy()
        for lo, hi in COUNT_BANDS:
            X[f"ratings: {lo}" if lo == hi else f"ratings: {lo}-{hi}"] = ((c >= lo) & (c <= hi)).astype(float)
        return _independent(X)
    if kind != "genre":
        raise ValueError("kind must be 'flat', 'genre' or 'catalogue'")
    f = films.set_index("film").reindex(index)
    genres = sorted({g for gs in films["genres"] for g in gs})
    cols = {"intercept": np.ones(len(index))}
    gl = f["genres"].map(lambda g: g if isinstance(g, list) else [])
    for g in genres:
        cols["genre: " + g] = gl.map(lambda gs, g=g: float(g in gs)).to_numpy()
    year = f["year"].to_numpy(dtype=float)
    decade = np.where(np.isnan(year), np.nan, np.clip(np.floor(year / 10) * 10, DECADES[0], DECADES[-1]))
    present = pd.Series(decade[~np.isnan(decade)]).value_counts()
    for d in sorted(present.index[1:]) if len(present) else []:  # the commonest decade is the baseline
        cols[f"decade: {int(d)}s"] = (decade == d).astype(float)
    cols["decade: unknown"] = np.isnan(year).astype(float)
    return _independent(pd.DataFrame(cols, index=index))


def _independent(X: pd.DataFrame) -> pd.DataFrame:
    """Drop columns no film has, and any column that is a combination of earlier ones (e.g. a last genre
    when every film has exactly one), so the predictors can be estimated."""
    X = X.loc[:, (X != 0).any(axis=0) | (X.columns == "intercept")]
    keep: list[str] = []
    for col in X.columns:
        trial = X[keep + [col]].to_numpy()
        gram = trial.T @ trial
        if np.linalg.matrix_rank(gram, tol=1e-8 * max(1.0, np.abs(gram).max())) == len(keep) + 1:
            keep.append(col)
    return X[keep]


def _fit_given_tau(X: np.ndarray, ybar: np.ndarray, samp_var: np.ndarray, tau: float):
    """Weighted least squares for gamma at a given tau; returns gamma, its covariance and log p(tau | y)
    up to a constant (flat priors on gamma and tau)."""
    V = samp_var + tau ** 2
    w = 1.0 / V
    XtWX = X.T @ (X * w[:, None])
    XtWy = X.T @ (w * ybar)
    cov = np.linalg.inv(XtWX)
    gamma = cov @ XtWy
    resid = ybar - X @ gamma
    sign, logdet = np.linalg.slogdet(XtWX)
    logpost = -0.5 * logdet - 0.5 * np.sum(np.log(V)) - 0.5 * np.sum(resid ** 2 * w)
    return gamma, cov, logpost


@dataclass
class Prior:
    """What all films together say about any one film before its own ratings are seen."""
    columns: list[str]
    sigma2: float                # spread of single ratings within a film (variance)
    tau_grid: np.ndarray         # values of tau considered
    tau_post: np.ndarray         # posterior probability of each value
    gammas: np.ndarray           # gamma at each tau on the grid (rows)
    covs: np.ndarray             # covariance of gamma at each tau on the grid

    @property
    def tau(self) -> float:
        """Posterior mean of tau."""
        return float(np.sum(self.tau_grid * self.tau_post))

    @property
    def gamma(self) -> pd.Series:
        """gamma at the posterior mean of tau's grid weights."""
        return pd.Series(self.tau_post @ self.gammas, index=self.columns)

    @property
    def kappa(self) -> float:
        """sigma^2 / tau^2: how many ratings' worth of weight the prediction carries."""
        return self.sigma2 / self.tau ** 2

    def tau_interval(self, level: float = 0.9) -> tuple[float, float]:
        cdf = np.cumsum(self.tau_post)
        lo = self.tau_grid[np.searchsorted(cdf, (1 - level) / 2)]
        hi = self.tau_grid[min(np.searchsorted(cdf, (1 + level) / 2), len(cdf) - 1)]
        return float(lo), float(hi)


def fit_prior(summary: pd.DataFrame, X: pd.DataFrame | None = None, sigma2: float | None = None,
              tau_max: float = 2.0, n_grid: int = 400) -> Prior:
    """Estimate gamma and tau from many films' averages.

    summary needs columns n and mean (and var, if sigma2 is not given); X is from design() and defaults to
    an intercept only. tau's grid runs from tau_max / n_grid to tau_max stars.
    """
    if X is None:
        X = pd.DataFrame({"intercept": 1.0}, index=summary.index)
    if not X.index.equals(summary.index):
        raise ValueError("X must have one row per film in summary, in the same order")
    X = _independent(X)  # e.g. a decade none of these films is from
    if len(summary) <= X.shape[1]:
        raise ValueError("need more films than predictors")
    sigma2 = pooled_sigma2(summary) if sigma2 is None else float(sigma2)
    Xa = X.to_numpy(dtype=float)
    ybar = summary["mean"].to_numpy(dtype=float)
    samp_var = sigma2 / summary["n"].to_numpy(dtype=float)
    grid = np.linspace(tau_max / n_grid, tau_max, n_grid)
    fits = [_fit_given_tau(Xa, ybar, samp_var, t) for t in grid]
    logpost = np.array([f[2] for f in fits])
    post = np.exp(logpost - logpost.max())
    post /= post.sum()
    if post[-1] > 1e-3:
        raise ValueError("tau's posterior reaches the top of the grid; raise tau_max")
    return Prior(columns=list(X.columns), sigma2=sigma2, tau_grid=grid, tau_post=post,
                 gammas=np.array([f[0] for f in fits]), covs=np.array([f[1] for f in fits]))


@dataclass
class ShapePrior:
    """A prior whose shape is learned from the data rather than assumed to be a bell curve.

    A film's true score is its prediction x_i' gamma plus a deviation d. The distribution of d is a set of
    probabilities on a fine grid of possible deviations (support), estimated from all films' averages.
    """
    columns: list[str]
    sigma2: float
    gamma: pd.Series
    support: np.ndarray
    weights: np.ndarray
    iterations: int

    def quantile(self, q: float) -> float:
        return float(np.interp(q, np.cumsum(self.weights), self.support))


def fit_shape_prior(summary: pd.DataFrame, X: pd.DataFrame | None = None, sigma2: float | None = None,
                    gamma: pd.Series | None = None, n_support: int = 300, max_iter: int = 500,
                    tol: float = 1e-7) -> ShapePrior:
    """Estimate the distribution of true deviations from each film's prediction.

    This is the nonparametric maximum likelihood estimate of a mixing distribution (Kiefer and Wolfowitz,
    1956), found by the EM algorithm of Laird (1978) on a fixed grid: each film's average is its true score
    plus noise with variance sigma^2 / n, and the weights on the grid are the mixture that makes all films'
    averages most probable. gamma defaults to the normal model's estimate (fit_prior).
    """
    if X is None:
        X = pd.DataFrame({"intercept": 1.0}, index=summary.index)
    X = _independent(X)
    sigma2 = pooled_sigma2(summary) if sigma2 is None else float(sigma2)
    if gamma is None:
        gamma = fit_prior(summary, X, sigma2=sigma2).gamma
    pred = X.to_numpy(dtype=float) @ gamma.reindex(X.columns).fillna(0.0).to_numpy()
    resid = summary["mean"].to_numpy(dtype=float) - pred
    s = np.sqrt(sigma2 / summary["n"].to_numpy(dtype=float))
    support = np.linspace(min(resid.min(), -1.0) - 0.5, max(resid.max(), 1.0) + 0.5, n_support)
    like = _likelihood(resid, s, support)
    w = np.full(n_support, 1.0 / n_support)
    for it in range(1, max_iter + 1):
        mix = like @ w
        w_new = w * (like.T @ (1.0 / mix)) / len(resid)
        if np.abs(w_new - w).max() < tol:
            w = w_new
            break
        w = w_new
    return ShapePrior(columns=list(X.columns), sigma2=sigma2, gamma=gamma.reindex(X.columns).fillna(0.0),
                      support=support, weights=w / w.sum(), iterations=it)


def _likelihood(resid: np.ndarray, s: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Normal likelihood of each film's residual (rows) under each grid deviation (columns)."""
    z = (resid[:, None] - support[None, :]) / s[:, None]
    return np.exp(-0.5 * z ** 2) / s[:, None]


def posterior_shape(summary: pd.DataFrame, prior: ShapePrior, X: pd.DataFrame | None = None,
                    level: float = 0.9, lower_q: float = 0.1, chunk: int = 20000) -> pd.DataFrame:
    """Each film's posterior under a ShapePrior, computed exactly on the grid (no normal approximation).

    Same columns as posterior(). Films are processed in chunks to keep memory down.
    """
    if X is None:
        X = pd.DataFrame({"intercept": 1.0}, index=summary.index)
    X = X.reindex(columns=prior.columns, fill_value=0.0)
    pred = X.to_numpy(dtype=float) @ prior.gamma.to_numpy()
    ybar = summary["mean"].to_numpy(dtype=float)
    n = summary["n"].to_numpy(dtype=float)
    s = np.sqrt(prior.sigma2 / n)
    out = {k: np.empty(len(n)) for k in ("score", "sd", "low", "high", "lower_bound")}
    qs = {"low": (1 - level) / 2, "high": (1 + level) / 2, "lower_bound": lower_q}
    for start in range(0, len(n), chunk):
        sl = slice(start, start + chunk)
        post = _likelihood(ybar[sl] - pred[sl], s[sl], prior.support) * prior.weights[None, :]
        post /= post.sum(axis=1, keepdims=True)
        mean_d = post @ prior.support
        out["score"][sl] = pred[sl] + mean_d
        out["sd"][sl] = np.sqrt(np.maximum(post @ prior.support ** 2 - mean_d ** 2, 0))
        cdf = np.cumsum(post, axis=1)
        for key, q in qs.items():
            k = (cdf < q).sum(axis=1).clip(max=len(prior.support) - 1)
            out[key][sl] = pred[sl] + prior.support[k]
    weight = np.where(np.abs(ybar - pred) > 1e-9, (out["score"] - pred) / np.where(
        np.abs(ybar - pred) > 1e-9, ybar - pred, 1.0), np.nan)
    return pd.DataFrame({"n": summary["n"].to_numpy(), "raw": ybar, "score": out["score"], "sd": out["sd"],
                         "weight": weight, "low": out["low"], "high": out["high"],
                         "lower_bound": out["lower_bound"]}, index=summary.index)


def posterior(summary: pd.DataFrame, prior: Prior, X: pd.DataFrame | None = None, level: float = 0.9,
              lower_q: float = 0.1) -> pd.DataFrame:
    """Each film's posterior: mean, sd, the weight on its own ratings, an interval and a lower bound.

    Averages over tau's grid posterior (only grid points with non-negligible probability are used) and
    includes the uncertainty in gamma. The interval and lower bound treat the posterior as normal.
    """
    if X is None:
        X = pd.DataFrame({"intercept": 1.0}, index=summary.index)
    if list(X.columns) != prior.columns:
        X = X.reindex(columns=prior.columns, fill_value=0.0)
    Xa = X.to_numpy(dtype=float)
    ybar = summary["mean"].to_numpy(dtype=float)
    n = summary["n"].to_numpy(dtype=float)
    samp_var = prior.sigma2 / n
    keep = np.flatnonzero(prior.tau_post > 1e-6 * prior.tau_post.max())
    p = prior.tau_post[keep] / prior.tau_post[keep].sum()
    m1 = np.zeros(len(n))
    m2 = np.zeros(len(n))
    wbar = np.zeros(len(n))
    for k, pk in zip(keep, p):
        tau2 = prior.tau_grid[k] ** 2
        pred = Xa @ prior.gammas[k]
        pred_var = np.einsum("ij,jk,ik->i", Xa, prior.covs[k], Xa)
        w = tau2 / (tau2 + samp_var)
        mean = w * ybar + (1 - w) * pred
        var = w * samp_var + (1 - w) ** 2 * pred_var
        m1 += pk * mean
        m2 += pk * (var + mean ** 2)
        wbar += pk * w
    sd = np.sqrt(np.maximum(m2 - m1 ** 2, 0))
    z = ndtri(0.5 + level / 2)
    return pd.DataFrame({
        "n": summary["n"].to_numpy(),
        "raw": ybar,
        "score": m1,
        "sd": sd,
        "weight": wbar,
        "low": m1 - z * sd,
        "high": m1 + z * sd,
        "lower_bound": m1 + ndtri(lower_q) * sd,
    }, index=summary.index)
