import zipfile

import numpy as np
import pandas as pd
import pytest

from filmscore.data import check_ratings, load_films, load_ratings, summarise


def test_summarise_matches_pandas(catalogue):
    s = summarise(catalogue.ratings)
    g = catalogue.ratings.groupby("film")["rating"].agg(["size", "mean", "var"])
    assert (s["n"].to_numpy() == g["size"].to_numpy()).all()
    assert np.allclose(s["mean"], g["mean"])
    both = g["var"].notna()
    assert np.allclose(s.loc[both, "var"], g.loc[both, "var"], atol=1e-6)
    assert s.loc[s["n"] == 1, "var"].isna().all()


def _write_movielens(folder):
    folder.mkdir()
    (folder / "ratings.csv").write_text("userId,movieId,rating,timestamp\n1,10,4.0,0\n2,10,3.5,0\n1,20,5.0,0\n")
    (folder / "movies.csv").write_text(
        'movieId,title,genres\n10,"Thing, The (1982)",Horror|Sci-Fi\n20,Untitled,(no genres listed)\n')


def test_load_from_folder_and_zip(tmp_path):
    folder = tmp_path / "ml-test"
    _write_movielens(folder)
    zpath = tmp_path / "ml-test.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for f in folder.iterdir():
            z.write(f, f"ml-test/{f.name}")
    for source in (folder, zpath):
        r = load_ratings(source)
        assert list(r.columns) == ["user", "film", "rating"] and len(r) == 3
        films = load_films(source)
        thing = films.set_index("film").loc[10]
        assert thing["title"] == "Thing, The" and thing["year"] == 1982 and thing["genres"] == ["Horror", "Sci-Fi"]
        untitled = films.set_index("film").loc[20]
        assert np.isnan(untitled["year"]) and untitled["genres"] == []


def test_check_ratings_rejects_bad_values():
    with pytest.raises(ValueError):
        check_ratings(pd.DataFrame({"user": [1], "film": [1], "rating": [6.0]}))
    with pytest.raises(ValueError):
        check_ratings(pd.DataFrame({"user": [1], "film": [1]}))
