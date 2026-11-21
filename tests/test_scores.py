import numpy as np
import pandas as pd
import pytest

from filmscore.data import summarise
from filmscore.scores import (design, fit_prior, fit_shape_prior, imdb_score, pooled_sigma2, posterior,
                              posterior_shape, raw_score)


def test_imdb_formula():
    s = pd.DataFrame({"n": [0, 10, 90], "mean": [5.0, 4.0, 3.0]})
    out = imdb_score(s, m=10, overall=3.5)
    assert out.tolist() == pytest.approx([3.5, 3.75, 3.05])
    assert raw_score(s).tolist() == [5.0, 4.0, 3.0]


def test_pooled_sigma2_weights_by_degrees_of_freedom():
    s = pd.DataFrame({"n": [1, 3, 11], "mean": [3, 3, 3], "var": [np.nan, 1.0, 2.0]})
    assert pooled_sigma2(s) == pytest.approx((2 * 1.0 + 10 * 2.0) / 12)


def test_fit_prior_recovers_tau(plain_catalogue):
    cat = plain_catalogue
    s = summarise(cat.ratings)
    X = design(cat.films, s.index, "genre")
    prior = fit_prior(s, X)
    true_resid = cat.true_score - X.to_numpy() @ np.linalg.lstsq(X.to_numpy(), cat.true_score.to_numpy(),
                                                                 rcond=None)[0]
    assert prior.tau == pytest.approx(true_resid.std(), rel=0.15)
    lo, hi = prior.tau_interval()
    assert lo < prior.tau < hi
    # every synthetic film has exactly one genre, so the last (Horror) is dropped and becomes the baseline
    assert "genre: Horror" not in prior.columns
    assert prior.gamma["genre: Documentary"] == pytest.approx(0.4 - (-0.3), abs=0.1)
    assert prior.gamma["genre: Drama"] == pytest.approx(0.2 - (-0.3), abs=0.1)


def test_posterior_is_a_weighted_average():
    s = pd.DataFrame({"n": [1, 4, 400], "mean": [5.0, 4.0, 2.0], "var": [np.nan, 1.0, 1.0]},
                     index=pd.Index([1, 2, 3], name="film"))
    many = pd.DataFrame({"n": np.full(500, 50), "mean": np.random.default_rng(0).normal(3, 0.5, 500),
                         "var": 1.0}, index=pd.Index(np.arange(10, 510), name="film"))
    prior = fit_prior(many, sigma2=1.0)
    p = posterior(s, prior)
    kappa = prior.kappa
    expected_w = s["n"] / (s["n"] + kappa)
    assert p["weight"].to_numpy() == pytest.approx(expected_w.to_numpy(), rel=0.05)
    centre = prior.gamma["intercept"]
    assert ((p["score"] - centre) * (s["mean"] - centre) >= 0).all()     # moves towards its own average
    assert (np.abs(p["score"] - centre) <= np.abs(s["mean"] - centre)).all()  # but never past it
    assert (p["lower_bound"] < p["score"]).all() and (p["low"] < p["lower_bound"]).all()
    assert p["sd"].is_monotonic_decreasing  # more ratings, less uncertainty


def test_posterior_intervals_cover_truth(plain_catalogue):
    cat = plain_catalogue
    s = summarise(cat.ratings)
    X = design(cat.films, s.index, "genre")
    p = posterior(s, fit_prior(s, X), X)
    truth = cat.true_score.loc[s.index]
    covered = ((truth >= p["low"]) & (truth <= p["high"])).mean()
    assert 0.86 <= covered <= 0.94
    assert np.sqrt(((p["score"] - truth) ** 2).mean()) < np.sqrt(((s["mean"] - truth) ** 2).mean())


def test_shape_prior_agrees_with_normal_when_truth_is_normal(plain_catalogue):
    cat = plain_catalogue
    s = summarise(cat.ratings)
    X = design(cat.films, s.index, "genre")
    normal = fit_prior(s, X)
    shape = fit_shape_prior(s, X, sigma2=normal.sigma2, gamma=normal.gamma)
    assert shape.weights.sum() == pytest.approx(1.0)
    a = posterior(s, normal, X)
    b = posterior_shape(s, shape, X)
    assert np.corrcoef(a["score"], b["score"])[0, 1] > 0.99
    assert (b["low"] <= b["score"]).all() and (b["score"] <= b["high"]).all()


def test_design_kinds(catalogue):
    s = summarise(catalogue.ratings)
    assert list(design(catalogue.films, s.index, "flat").columns) == ["intercept"]
    g = design(catalogue.films, s.index, "genre")
    assert {"genre: Drama", "genre: Documentary"} <= set(g.columns)
    assert np.linalg.matrix_rank(g.to_numpy()) == g.shape[1]
    c = design(catalogue.films, s.index, "catalogue", counts=s["n"])
    assert "ratings: 1" in c.columns and (c.filter(like="ratings:").sum(axis=1) <= 1).all()
    with pytest.raises(ValueError):
        design(catalogue.films, s.index, "catalogue")
