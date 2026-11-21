import numpy as np
import pytest

from filmscore.data import summarise
from filmscore.evaluate import Subsampler, rater_thin_data, ranking, reference_films, thin_data


@pytest.fixture(scope="module")
def setup(catalogue):
    s = summarise(catalogue.ratings)
    ref = reference_films(s, min_n=40, folds=2, seed=0)
    return catalogue, s, ref


def test_reference_folds(setup):
    _, s, ref = setup
    assert (s.loc[ref.index, "n"] >= 40).all()
    assert abs((ref == 0).sum() - (ref == 1).sum()) <= 1


def test_subsamples_are_nested_and_the_right_size(setup):
    cat, s, ref = setup
    sampler = Subsampler(cat.ratings, ref.index, seed=3)
    s5 = sampler.summary(5)
    assert (s5["n"] == 5).all()
    full = sampler.summary(10 ** 6)
    assert np.allclose(full["mean"], s.loc[full.index, "mean"])
    # nested: the first 1 rating is one of the first 5
    s1 = sampler.summary(1)
    assert ((s1["mean"] >= 0.5) & (s1["mean"] <= 5)).all()
    per_film = np.minimum(np.arange(len(full)) % 7 + 1, full["n"].to_numpy())
    assert (sampler.summary(per_film)["n"].to_numpy() == per_film).all()


def test_pooling_beats_the_raw_average_with_few_ratings(setup):
    cat, s, ref = setup
    res = thin_data(cat.ratings, cat.films, s, ref, ns=(1, 3, 30), ms=(5,), replicates=2)
    rmse = res.groupby(["n", "method"])["rmse"].mean().unstack()
    assert rmse.loc[1, "pooled (genre)"] < 0.7 * rmse.loc[1, "raw"]
    assert rmse.loc[30, "pooled (genre)"] <= rmse.loc[30, "raw"] + 0.01
    cov = res.dropna(subset=["coverage"]).groupby("method")["coverage"].mean()
    assert ((cov > 0.8) & (cov < 0.97)).all()


def test_ranking_by_lower_bound_picks_better_films_than_the_raw_average(setup):
    cat, s, ref = setup
    res = ranking(cat.ratings, cat.films, s, ref, count_pool=np.array([1, 2, 3, 5, 10, 40]), tops=(20,),
                  replicates=4, min_ratings=5, ms=(10,))
    by = res.groupby("method")["mean_truth"].mean()
    assert by["pooled (genre), lower bound"] > by["raw average"]
    gaps = res.groupby("method")["gap"].mean()
    assert gaps["pooled (genre), lower bound"] < gaps["raw average"]


def test_rater_adjustment_helps_when_raters_differ(setup):
    cat, s, ref = setup
    res, info = rater_thin_data(cat.ratings, cat.films, s, ref, ns=(3, 10), replicates=1, max_iter=30)
    rmse = res.groupby(["n", "method"])["rmse"].mean().unstack()
    assert (rmse["pooled (genre), rater-adjusted"] < rmse["pooled (genre)"]).all()
    assert info["tau_user"] > 0.2
