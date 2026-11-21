"""Score every film in MovieLens 32M and write the small table the dashboard reads.

For the whole catalogue the prior uses genre, decade and the number of ratings (films few people rate tend
to be rated lower). Writes:

  app_data/films.parquet   one row per film: title, year, genres, ratings, raw average, pooled score,
                           interval, lower bound, and the same allowing for who rated it
  app_data/model.json      the fitted model's headline numbers
  results/catalogue_top.md the top 15 films under each way of ranking

Usage:  python scripts/build_app_data.py --data data/ml-32m.zip
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filmscore import fit_rater_model, summarise  # noqa: E402
from filmscore.scores import design, fit_prior, posterior  # noqa: E402
from run_evaluation import load  # noqa: E402

LOWER_Q = 0.1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "ml-32m.zip")
    args = ap.parse_args()
    t0 = time.time()
    ratings, films = load(args.data)
    summary = summarise(ratings)
    X = design(films, summary.index, "catalogue", counts=summary["n"])
    prior = fit_prior(summary, X)
    post = posterior(summary, prior, X, lower_q=LOWER_Q)
    gamma = prior.gamma
    prediction = X.reindex(columns=gamma.index, fill_value=0.0).to_numpy() @ gamma.to_numpy()
    print(f"catalogue prior: tau {prior.tau:.3f}, kappa {prior.kappa:.2f} ({time.time() - t0:.0f}s)")

    model = fit_rater_model(ratings)
    adjusted = summarise(model.adjust(ratings)).loc[summary.index]
    prior_adj = fit_prior(adjusted, X)
    post_adj = posterior(adjusted, prior_adj, X, lower_q=LOWER_Q)
    print(f"rater model: {model.iterations} iterations, tau_user {model.tau_user:.3f} ({time.time() - t0:.0f}s)")

    meta = films.set_index("film").reindex(summary.index)
    table = pd.DataFrame({
        "title": meta["title"].fillna("(unknown)"),
        "year": meta["year"].astype("Float32"),
        "genres": meta["genres"].map(lambda g: "|".join(g) if isinstance(g, list) else ""),
        "n": summary["n"].astype(np.int32),
        "raw": summary["mean"].astype(np.float32),
        "prediction": prediction.astype(np.float32),
        "score": post["score"].astype(np.float32),
        "sd": post["sd"].astype(np.float32),
        "low": post["low"].astype(np.float32),
        "high": post["high"].astype(np.float32),
        "lower_bound": post["lower_bound"].astype(np.float32),
        "weight": post["weight"].astype(np.float32),
        "adj_raw": adjusted["mean"].astype(np.float32),
        "adj_score": post_adj["score"].astype(np.float32),
        "adj_low": post_adj["low"].astype(np.float32),
        "adj_high": post_adj["high"].astype(np.float32),
        "adj_lower_bound": post_adj["lower_bound"].astype(np.float32),
    }, index=summary.index)
    out = ROOT / "app_data"
    out.mkdir(exist_ok=True)
    table.reset_index().to_parquet(out / "films.parquet", index=False, compression="zstd")
    (out / "model.json").write_text(json.dumps({
        "ratings": int(len(ratings)), "films": int(len(summary)), "users": int(ratings["user"].nunique()),
        "overall_mean": float(ratings["rating"].mean()), "film_average_mean": float(summary["mean"].mean()),
        "sigma": float(np.sqrt(prior.sigma2)), "tau": prior.tau, "kappa": prior.kappa,
        "tau_interval": prior.tau_interval(), "gamma": {k: round(float(v), 4) for k, v in gamma.items()},
        "lower_q": LOWER_Q,
        "rater_model": {"mu": model.mu, "tau_film": model.tau_film, "tau_user": model.tau_user,
                        "sigma": float(np.sqrt(model.sigma2)), "iterations": model.iterations,
                        "sigma_adjusted": float(np.sqrt(prior_adj.sigma2)), "tau_adjusted": prior_adj.tau,
                        "kappa_adjusted": prior_adj.kappa},
    }, indent=2))

    def top(col, k=15, min_n=1, prefix=""):
        t = table[table["n"] >= min_n].sort_values([col, "n"], ascending=[False, False]).head(k)
        t = t.assign(film=t["title"] + t["year"].map(lambda y: "" if pd.isna(y) else f" ({int(y)})"))
        return t[["film", "n", prefix + "raw", prefix + "score", prefix + "lower_bound"]].rename(columns={
            "film": "Film", "n": "Ratings", prefix + "raw": "Raw average", prefix + "score": "Pooled score",
            prefix + "lower_bound": "Lower bound"}).round(2).to_markdown(index=False)

    sections = [("Raw average, any number of ratings", top("raw")),
                ("Raw average, at least 100 ratings", top("raw", min_n=100)),
                ("Pooled score", top("score")),
                (f"Lower bound (we are {100 - 100 * LOWER_Q:.0f}% sure the film is at least this good)",
                 top("lower_bound")),
                ("Lower bound, allowing for who rated each film (columns are rater-adjusted)",
                 top("adj_lower_bound", prefix="adj_"))]
    (ROOT / "results" / "catalogue_top.md").write_text(
        "Top 15 films in MovieLens 32M under each way of ranking.\n\n"
        + "\n\n".join(f"## {name}\n\n{body}" for name, body in sections) + "\n")
    print(f"done ({time.time() - t0:.0f}s); films.parquet is "
          f"{(out / 'films.parquet').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
