"""Loading MovieLens ratings and summarising them film by film.

MovieLens 32M (GroupLens, University of Minnesota) holds 32 million ratings of 87,585 films by 200,948 users,
on a scale of 0.5 to 5 stars in half-star steps. It is downloaded from GroupLens rather than stored in this
repository: https://files.grouplens.org/datasets/movielens/ml-32m.zip

Ratings are kept as three compact columns: user (int32), film (int32, the MovieLens movieId) and rating
(float32), which is about 400 MB for the full data set.
"""

from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def download(dest: str | Path, url: str = MOVIELENS_URL) -> Path:
    """Download a MovieLens zip to dest (a file path), unless it is already there."""
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    return dest


def _open(source: str | Path, name: str):
    """Open name (e.g. 'ratings.csv') from a MovieLens zip or an unzipped folder."""
    source = Path(source)
    if source.is_dir():
        matches = sorted(source.rglob(name))
        if not matches:
            raise FileNotFoundError(f"{name} not found under {source}")
        return open(matches[0], "rb")
    archive = zipfile.ZipFile(source)
    members = [m for m in archive.namelist() if m.endswith("/" + name) or m == name]
    if not members:
        raise FileNotFoundError(f"{name} not found in {source}")
    return io.BytesIO(archive.read(members[0]))


def load_ratings(source: str | Path) -> pd.DataFrame:
    """Ratings as a DataFrame with columns user, film, rating.

    source is a MovieLens zip, an unzipped MovieLens folder, or a parquet file written by save_parquet().
    """
    source = Path(source)
    if source.suffix == ".parquet":
        return pd.read_parquet(source)
    import pyarrow as pa
    import pyarrow.csv as pc

    with _open(source, "ratings.csv") as handle:
        table = pc.read_csv(handle, convert_options=pc.ConvertOptions(
            include_columns=["userId", "movieId", "rating"],
            column_types={"userId": pa.int32(), "movieId": pa.int32(), "rating": pa.float32()}))
    ratings = table.to_pandas().rename(columns={"userId": "user", "movieId": "film"})
    check_ratings(ratings)
    return ratings


def check_ratings(ratings: pd.DataFrame) -> None:
    """Stop with a clear message if the ratings are not what the rest of the code assumes."""
    missing = {"user", "film", "rating"} - set(ratings.columns)
    if missing:
        raise ValueError(f"ratings are missing columns: {sorted(missing)}")
    if ratings[["user", "film", "rating"]].isna().any().any():
        raise ValueError("ratings contain missing values")
    r = ratings["rating"].to_numpy()
    if r.min() < 0.5 or r.max() > 5:
        raise ValueError("ratings must lie between 0.5 and 5 stars")


def load_films(source: str | Path) -> pd.DataFrame:
    """One row per film: film (movieId), title, year (from the title, may be missing) and genres (a list)."""
    with _open(source, "movies.csv") as handle:
        films = pd.read_csv(handle, dtype={"movieId": np.int32, "title": str, "genres": str})
    films = films.rename(columns={"movieId": "film"})
    films["year"] = films["title"].map(_year)
    films["title"] = films["title"].str.replace(r"\s*\(\d{4}[^)]*\)\s*$", "", regex=True).str.strip()
    films["genres"] = films["genres"].map(
        lambda g: [] if g in ("(no genres listed)", "") or pd.isna(g) else g.split("|"))
    return films


def _year(title: str) -> float:
    match = re.search(r"\((\d{4})[^)]*\)\s*$", str(title))
    return float(match.group(1)) if match else np.nan


def save_parquet(ratings: pd.DataFrame, path: str | Path) -> Path:
    """Cache the ratings as parquet, which loads in about a second."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ratings.to_parquet(path, index=False)
    return path


def summarise(ratings: pd.DataFrame) -> pd.DataFrame:
    """Per-film count, mean and sample variance of ratings, indexed by film.

    var is the sample variance (divisor n - 1); it is missing for films with a single rating.
    """
    film_ids, idx = np.unique(ratings["film"].to_numpy(), return_inverse=True)
    r = ratings["rating"].to_numpy(dtype=np.float64)
    n = np.bincount(idx)
    total = np.bincount(idx, weights=r)
    total_sq = np.bincount(idx, weights=r * r)
    mean = total / n
    with np.errstate(invalid="ignore", divide="ignore"):
        var = np.where(n > 1, (total_sq - n * mean ** 2) / (n - 1), np.nan)
    var = np.where(var < 0, 0.0, var)  # rounding can push a zero variance just below zero
    return pd.DataFrame({"n": n, "mean": mean, "var": var}, index=pd.Index(film_ids, name="film"))
