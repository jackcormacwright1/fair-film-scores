"""Allowing for who rated a film: some people rate everything generously, others harshly.

The model adds a rater effect, the equivalent of the surgeon effect in the theatre scheduler:

    rating of film i by user u:  r_ui = mu + a_i + b_u + e_ui
    film effects:                 a_i ~ Normal(0, tau_film^2)
    user effects:                 b_u ~ Normal(0, tau_user^2)      (b_u > 0: rates everything higher)
    noise:                        e_ui ~ Normal(0, sigma^2)

It is fitted by alternating between the films and the users: given the user effects, each film effect is
its users' average residual shrunk towards zero, and vice versa. For a model this simple that converges to
the joint posterior mode, which for normal distributions is also the posterior mean. After each pass the
film and user effects are re-centred on zero, with their averages moved into mu: they average zero at the
solution, and without this step the alternating updates take hundreds of passes to settle the overall
level. The variances are re-estimated after each pass (an approximate EM step that ignores the small
correlation between a film's and its users' estimates).

A film's rater-adjusted ratings are r_ui - b_u: each rating with the rater's habit taken out. Averaging and
shrinking those (with filmscore.scores) gives a score for how an average rater would judge the film.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RaterModel:
    mu: float
    film_effect: pd.Series      # a_i, indexed by film
    user_effect: pd.Series      # b_u, indexed by user
    tau_film: float
    tau_user: float
    sigma2: float
    iterations: int

    def adjust(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """ratings with each rating minus its rater's effect (users the model has not seen get 0)."""
        b = self.user_effect.reindex(ratings["user"].to_numpy()).fillna(0.0).to_numpy(dtype=np.float32)
        out = ratings[["user", "film"]].copy()
        out["rating"] = ratings["rating"].to_numpy(dtype=np.float32) - b
        return out


def fit_rater_model(ratings: pd.DataFrame, max_iter: int = 50, tol: float = 1e-5) -> RaterModel:
    """Fit mu, the film and user effects and the three variances; stops when the effects change by
    less than tol stars between passes."""
    films, fi = np.unique(ratings["film"].to_numpy(), return_inverse=True)
    users, ui = np.unique(ratings["user"].to_numpy(), return_inverse=True)
    r = ratings["rating"].to_numpy(dtype=np.float64)
    n_f = np.bincount(fi).astype(float)
    n_u = np.bincount(ui).astype(float)
    mu = r.mean()
    a = np.zeros(len(films))
    b = np.zeros(len(users))
    tau_f2, tau_u2, sigma2 = 0.25, 0.25, float(r.var())
    for it in range(1, max_iter + 1):
        # films given users
        prec_f = n_f / sigma2 + 1 / tau_f2
        a_new = np.bincount(fi, weights=r - mu - b[ui], minlength=len(films)) / sigma2 / prec_f
        # users given films
        prec_u = n_u / sigma2 + 1 / tau_u2
        b_new = np.bincount(ui, weights=r - mu - a_new[fi], minlength=len(users)) / sigma2 / prec_u
        mu = float(np.mean(r - a_new[fi] - b_new[ui]))
        # At the solution the film effects and the user effects each average exactly zero (their priors are
        # centred there and mu is free), so move their averages into mu. Plain alternating updates only
        # creep along this direction, which made convergence very slow.
        mu += a_new.mean() + b_new.mean()
        a_new -= a_new.mean()
        b_new -= b_new.mean()
        # variances (approximate EM)
        v_f, v_u = 1 / prec_f, 1 / prec_u
        tau_f2 = float(np.mean(a_new ** 2 + v_f))
        tau_u2 = float(np.mean(b_new ** 2 + v_u))
        resid = r - mu - a_new[fi] - b_new[ui]
        sigma2 = float((np.sum(resid ** 2) + np.sum(n_f * v_f) + np.sum(n_u * v_u)) / len(r))
        change = max(np.abs(a_new - a).max(), np.abs(b_new - b).max())
        a, b = a_new, b_new
        if change < tol:
            break
    return RaterModel(mu=mu, film_effect=pd.Series(a, index=pd.Index(films, name="film")),
                      user_effect=pd.Series(b, index=pd.Index(users, name="user")),
                      tau_film=float(np.sqrt(tau_f2)), tau_user=float(np.sqrt(tau_u2)), sigma2=sigma2,
                      iterations=it)
