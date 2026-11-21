import numpy as np
import pytest

from filmscore.raters import fit_rater_model


def test_rater_model_recovers_user_effects(catalogue):
    model = fit_rater_model(catalogue.ratings)
    common = model.user_effect.index.intersection(catalogue.user_effect.index)
    est = model.user_effect.loc[common]
    true = catalogue.user_effect.loc[common]
    assert np.corrcoef(est, true)[0, 1] > 0.85
    assert model.tau_user == pytest.approx(true.std(), rel=0.25)
    film_est = model.mu + model.film_effect
    assert np.corrcoef(film_est, catalogue.true_score.loc[film_est.index])[0, 1] > 0.85


def test_adjust_removes_each_raters_effect(catalogue):
    model = fit_rater_model(catalogue.ratings, max_iter=5)
    r = catalogue.ratings.head(50)
    adjusted = model.adjust(r)
    expected = r["rating"].to_numpy() - model.user_effect.loc[r["user"]].to_numpy()
    assert adjusted["rating"].to_numpy() == pytest.approx(expected, abs=1e-5)
