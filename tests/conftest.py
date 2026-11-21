import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filmscore.synthetic import make_catalogue  # noqa: E402


@pytest.fixture(scope="session")
def catalogue():
    """Films with known true scores, rater effects and a wide spread of rating counts."""
    return make_catalogue(n_films=1500, n_users=2500, median_count=15, seed=1)


@pytest.fixture(scope="session")
def plain_catalogue():
    """The same, without rater effects, so the normal model of scores.py holds exactly."""
    return make_catalogue(n_films=3000, n_users=2500, median_count=10, seed=2, rater_effects=False)
