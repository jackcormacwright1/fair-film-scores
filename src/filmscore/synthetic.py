"""A small synthetic film catalogue with known true scores, for tests and demonstrations.

True film scores, rater habits and the number of ratings per film are drawn from the model in scores.py
and raters.py, so the right answers are known exactly. Ratings are rounded to half stars and kept between
0.5 and 5, as in MovieLens.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

GENRES = ["Comedy", "Drama", "Documentary", "Horror"]
GENRE_EFFECT = {"Comedy": -0.1, "Drama": 0.2, "Documentary": 0.4, "Horror": -0.3}


@dataclass
class Catalogue:
    ratings: pd.DataFrame       # user, film, rating
    films: pd.DataFrame         # film, title, year, genres
    true_score: pd.Series       # theta_i: the film's average with unlimited ratings (before rounding)
    user_effect: pd.Series      # b_u


def make_catalogue(n_films: int = 2000, n_users: int = 3000, mu: float = 3.4, tau: float = 0.45,
                   tau_user: float = 0.4, sigma: float = 0.75, median_count: float = 20, seed: int = 0,
                   rater_effects: bool = True) -> Catalogue:
    """Films with one genre each; each film's number of ratings is lognormal around median_count."""
    rng = np.random.default_rng(seed)
    film_ids = np.arange(1, n_films + 1, dtype=np.int32)
    genre = rng.choice(GENRES, size=n_films)
    year = rng.integers(1950, 2024, size=n_films).astype(float)
    theta = mu + np.array([GENRE_EFFECT[g] for g in genre]) + rng.normal(0, tau, n_films)
    b = rng.normal(0, tau_user, n_users) if rater_effects else np.zeros(n_users)
    counts = np.maximum(1, np.round(median_count * rng.lognormal(0, 1.2, n_films))).astype(int)
    counts = np.minimum(counts, n_users)
    film_col = np.repeat(film_ids, counts)
    user_col = np.concatenate([rng.choice(n_users, size=c, replace=False) for c in counts]).astype(np.int32)
    raw = theta[film_col - 1] + b[user_col] + rng.normal(0, sigma, len(film_col))
    rating = np.clip(np.round(raw * 2) / 2, 0.5, 5.0).astype(np.float32)
    ratings = pd.DataFrame({"user": user_col + 1, "film": film_col, "rating": rating})
    films = pd.DataFrame({"film": film_ids, "title": [f"Film {i}" for i in film_ids], "year": year,
                          "genres": [[g] for g in genre]})
    return Catalogue(ratings=ratings, films=films,
                     true_score=pd.Series(theta, index=pd.Index(film_ids, name="film")),
                     user_effect=pd.Series(b, index=pd.Index(np.arange(1, n_users + 1), name="user")))
