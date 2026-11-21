"""Testing scores against a known answer.

Films with many ratings (1,000 or more by default) have an average that is all but certain: their "truth".
Each film is then shown to a method with only a few of its ratings, chosen at random, and the method's
estimate is compared with the truth. Everything a method learns from other films (the prior, the IMDb
overall average, the rater effects) comes from a separate set of films, so a film never helps to judge
itself: the films are split into folds, and each fold is scored using what the other folds teach.

Two experiments:

* thin_data(): every film shown the same number of ratings n, for a range of n. Accuracy and calibration.
* ranking(): every film shown a number of ratings drawn from the real spread of rating counts (half of
  MovieLens films have five or fewer), then ranked. Are the films at the top really the best?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import summarise
from .scores import (design, fit_prior, fit_shape_prior, imdb_score, pooled_sigma2, posterior,
                     posterior_shape)


def reference_films(summary: pd.DataFrame, min_n: int = 1000, folds: int = 2, seed: int = 0) -> pd.Series:
    """Fold number (0 .. folds-1) for each film with at least min_n ratings, assigned at random."""
    ref = summary.index[summary["n"] >= min_n]
    if len(ref) < 10 * folds:
        raise ValueError("too few reference films; lower min_n")
    rng = np.random.default_rng(seed)
    fold = rng.permutation(np.arange(len(ref)) % folds)
    return pd.Series(fold, index=ref, name="fold")


class Subsampler:
    """Random subsets of each film's ratings, nested: the first n of a shuffled order, so the 5 ratings a
    film is shown are among the 10 it is shown at n = 10."""

    def __init__(self, ratings: pd.DataFrame, films: pd.Index, seed: int):
        sub = ratings[ratings["film"].isin(films)]
        rng = np.random.default_rng(seed)
        key = rng.random(len(sub))
        f = sub["film"].to_numpy()
        order = np.lexsort((key, f))
        self.film_ids, idx = np.unique(f[order], return_inverse=True)
        self.r = sub["rating"].to_numpy(dtype=np.float64)[order]
        starts = np.r_[0, np.flatnonzero(np.diff(idx)) + 1]
        self.pos = np.arange(len(idx)) - starts[idx]
        self.idx = idx
        self.count = np.bincount(idx)

    def summary(self, n) -> pd.DataFrame:
        """Each film's count and mean using its first n shuffled ratings; n is a number or one per film."""
        n = np.broadcast_to(np.asarray(n), self.count.shape)
        n = np.minimum(n, self.count)
        take = self.pos < n[self.idx]
        total = np.bincount(self.idx[take], weights=self.r[take], minlength=len(self.count))
        return pd.DataFrame({"n": n, "mean": total / n}, index=pd.Index(self.film_ids, name="film"))


def _truth(summary: pd.DataFrame, ref: pd.Series) -> pd.Series:
    return summary.loc[ref.index, "mean"]


def _designs(films: pd.DataFrame, ref: pd.Series, kinds=("flat", "genre")) -> dict[str, pd.DataFrame]:
    """Predictors for all reference films, built once so every fold codes genres and decades alike."""
    return {kind: design(films, ref.index, kind) for kind in kinds}


def _fold_priors(summary: pd.DataFrame, X: dict[str, pd.DataFrame], train: pd.Index):
    """Priors fitted on the training films' full data, with their sigma^2 and overall average."""
    s = summary.loc[train]
    sigma2 = pooled_sigma2(s)
    out = {kind: fit_prior(s, Xk.loc[train], sigma2=sigma2) for kind, Xk in X.items()}
    return out, sigma2, float(s["mean"].mean())


def thin_data(ratings: pd.DataFrame, films: pd.DataFrame, summary: pd.DataFrame, ref: pd.Series,
              ns=(1, 2, 3, 5, 10, 20, 50, 100), ms=(1, 2, 5, 10, 25, 50, 100, 250, 1000),
              replicates: int = 5, seed: int = 1, truth: pd.Series | None = None) -> pd.DataFrame:
    """Error of each method when every reference film is shown n ratings.

    Returns one row per (replicate, n, method) with RMSE, mean absolute error, and (for methods that give
    one) the coverage and width of a 90% interval. truth defaults to each film's full-data average.
    """
    truth = _truth(summary, ref) if truth is None else truth.loc[ref.index]
    z = 1.6448536269514722
    rows = []
    X_all = _designs(films, ref)
    for fold in sorted(ref.unique()):
        test = ref.index[ref == fold]
        train = ref.index[ref != fold]
        priors, sigma2, overall = _fold_priors(summary, X_all, train)
        X = {k: Xk.loc[test] for k, Xk in X_all.items()}
        shape = fit_shape_prior(summary.loc[train], X_all["genre"].loc[train], sigma2=sigma2,
                                gamma=priors["genre"].gamma)
        for rep in range(replicates):
            sampler = Subsampler(ratings, test, seed=seed + 1000 * rep + fold)
            for n in ns:
                s = sampler.summary(n).loc[test]
                t = truth.loc[test].to_numpy()
                est = {"raw": (s["mean"].to_numpy(), np.sqrt(sigma2 / s["n"].to_numpy()))}
                for m in ms:
                    est[f"imdb m={m}"] = (imdb_score(s, m, overall).to_numpy(), None)
                for kind, prior in priors.items():
                    p = posterior(s, prior, X[kind])
                    est[f"pooled ({kind})"] = (p["score"].to_numpy(), p["sd"].to_numpy())
                p = posterior_shape(s, shape, X["genre"])
                est["pooled (genre), data-shaped prior"] = (p["score"].to_numpy(), p["sd"].to_numpy(),
                                                            p["low"].to_numpy(), p["high"].to_numpy())
                for name, values in est.items():
                    e, sd = values[0], values[1]
                    err = e - t
                    row = {"fold": fold, "replicate": rep, "n": n, "method": name, "films": len(t),
                           "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
                           "bias": float(np.mean(err))}
                    if len(values) == 4:  # an interval from the grid posterior, not symmetric
                        lo, hi = values[2], values[3]
                        row["coverage"] = float(np.mean((t >= lo) & (t <= hi)))
                        row["width"] = float(np.mean(hi - lo))
                    elif sd is not None:
                        row["coverage"] = float(np.mean(np.abs(err) <= z * sd))
                        row["width"] = float(np.mean(2 * z * sd))
                    rows.append(row)
    return pd.DataFrame(rows)


def ranking(ratings: pd.DataFrame, films: pd.DataFrame, summary: pd.DataFrame, ref: pd.Series,
            count_pool: np.ndarray, tops=(20, 100), replicates: int = 20, seed: int = 2,
            min_ratings: int = 20, ms=(25, 100), lower_q: float = 0.1) -> pd.DataFrame:
    """Rank reference films after showing each a realistic number of ratings, then check the top.

    count_pool: rating counts to draw from (e.g. every film's count in the full data). Each film's shown
    count is a random draw from it, capped at the film's own count. Methods: the raw average; the raw
    average among films with at least min_ratings ratings (the usual fix); the IMDb formula for each m; and
    the pooled score with a flat or a genre prior, ranked by its posterior mean or by its lower bound (the
    lower_q quantile of the posterior: "we are 90% sure the film is at least this good").

    Returns one row per (replicate, method, top): the mean of the scores the method showed for its top
    films and of their true scores, the gap between the two, the share truly in the best 10% of reference
    films, the share truly below their median, the worst true score, and the median number of ratings the
    picked films were shown.
    """
    truth = _truth(summary, ref)
    best10 = truth.quantile(0.9)
    median = truth.median()
    rng = np.random.default_rng(seed)
    X_all = _designs(films, ref)
    fold_priors = {fold: _fold_priors(summary, X_all, ref.index[ref != fold]) for fold in sorted(ref.unique())}
    rows = []
    for rep in range(replicates):
        scored = []
        for fold in sorted(ref.unique()):
            test = ref.index[ref == fold]
            priors, sigma2, overall = fold_priors[fold]
            sampler = Subsampler(ratings, test, seed=seed + 1000 * rep + fold)
            shown = rng.choice(count_pool, size=len(sampler.count))
            s = sampler.summary(shown).loc[test]
            cols = {"raw average": s["mean"],
                    f"raw average, {min_ratings}+ ratings": s["mean"].where(s["n"] >= min_ratings)}
            for m in ms:
                cols[f"IMDb formula, m = {m:g}"] = imdb_score(s, m, overall)
            for kind, prior in priors.items():
                p = posterior(s, prior, X_all[kind].loc[test], lower_q=lower_q)
                cols[f"pooled ({kind})"] = p["score"]
                cols[f"pooled ({kind}), lower bound"] = p["lower_bound"]
            frame = pd.DataFrame(cols)
            frame["shown"] = s["n"]
            scored.append(frame)
        scored = pd.concat(scored)
        jitter = pd.Series(rng.random(len(scored)) * 1e-9, index=scored.index)  # break ties at random
        for method in [c for c in scored.columns if c != "shown"]:
            ranked = (scored[method] + jitter).dropna().sort_values(ascending=False)
            for top in tops:
                chosen = ranked.index[:top]
                t = truth.loc[chosen]
                shown_score = scored.loc[chosen, method]
                rows.append({"replicate": rep, "method": method, "top": top,
                             "mean_shown": float(shown_score.mean()), "mean_truth": float(t.mean()),
                             "gap": float(shown_score.mean() - t.mean()),
                             "share_best10": float((t >= best10).mean()),
                             "share_below_median": float((t < median).mean()), "worst_truth": float(t.min()),
                             "median_ratings_shown": float(scored.loc[chosen, "shown"].median())})
    return pd.DataFrame(rows)


def rater_thin_data(ratings: pd.DataFrame, films: pd.DataFrame, summary: pd.DataFrame, ref: pd.Series,
                    ns=(1, 2, 3, 5, 10, 20, 50), replicates: int = 3, seed: int = 3, max_iter: int = 60,
                    full_model=None) -> tuple[pd.DataFrame, dict]:
    """How well does each method recover a film's rater-adjusted score from n ratings?

    The target here is different from thin_data(): the average an average rater would give, which is
    each rating minus its rater's habit, averaged over all of the film's ratings in the full data. The rater
    effects used for the n shown ratings come from a model fitted without any of the test films' ratings.

    Returns per-(fold, replicate, n, method) errors, and a dict of fitted values for the write-up.
    """
    from .raters import fit_rater_model

    full = fit_rater_model(ratings, max_iter=max_iter) if full_model is None else full_model
    adjusted_full = full.adjust(ratings)
    truth = summarise(adjusted_full).loc[ref.index, "mean"]
    info = {"tau_user": full.tau_user, "tau_film": full.tau_film, "sigma": float(np.sqrt(full.sigma2)),
            "iterations": full.iterations,
            "corr_raw_vs_adjusted_truth": float(np.corrcoef(truth, summary.loc[ref.index, "mean"])[0, 1]),
            "sd_raw_minus_adjusted_truth": float((summary.loc[ref.index, "mean"] - truth).std())}
    X_genre = design(films, ref.index, "genre")
    rows = []
    for fold in sorted(ref.unique()):
        test = ref.index[ref == fold]
        train = ref.index[ref != fold]
        keep = ~ratings["film"].isin(test)
        model = fit_rater_model(ratings[keep], max_iter=max_iter)
        info[f"iterations_fold{fold}"] = model.iterations
        adjusted = model.adjust(ratings)
        s_raw = summary.loc[train]
        s_adj = summarise(adjusted[adjusted["film"].isin(train)]).loc[train]
        X_train, X_test = X_genre.loc[train], X_genre.loc[test]
        prior_raw = fit_prior(s_raw, X_train, sigma2=pooled_sigma2(s_raw))
        prior_adj = fit_prior(s_adj, X_train, sigma2=pooled_sigma2(s_adj))
        info[f"sigma_raw_fold{fold}"] = float(np.sqrt(prior_raw.sigma2))
        info[f"sigma_adjusted_fold{fold}"] = float(np.sqrt(prior_adj.sigma2))
        for rep in range(replicates):
            sub_seed = seed + 1000 * rep + fold
            raw_sampler = Subsampler(ratings, test, seed=sub_seed)
            adj_sampler = Subsampler(adjusted, test, seed=sub_seed)  # same rows, same order: same subsets
            for n in ns:
                s1 = raw_sampler.summary(n).loc[test]
                s2 = adj_sampler.summary(n).loc[test]
                t = truth.loc[test].to_numpy()
                est = {"raw average": s1["mean"].to_numpy(),
                       "pooled (genre)": posterior(s1, prior_raw, X_test)["score"].to_numpy(),
                       "pooled (genre), rater-adjusted": posterior(s2, prior_adj, X_test)["score"].to_numpy()}
                for name, e in est.items():
                    err = e - t
                    rows.append({"fold": fold, "replicate": rep, "n": n, "method": name,
                                 "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err)))})
    return pd.DataFrame(rows), info
