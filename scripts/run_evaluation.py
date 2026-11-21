"""Run the known-answer tests on MovieLens 32M and write the results.

Films with at least 1,000 ratings are the reference films: their full-data average is the truth. Each is
then shown only a few of its ratings, and each method's estimate is compared with the truth. See
filmscore/evaluate.py for the design.

Writes, in results/:
  thin_data.csv, thin_data.md, thin_data.png, fixed_m.png   accuracy with n ratings
  ranking.csv, ranking.md                                    which films reach the top, and how good they are
  raters.csv, raters.md                                      allowing for who rated a film
  summary.json                                               headline numbers

Usage:  python scripts/run_evaluation.py --data data/ml-32m.zip [--replicates 5] [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filmscore import load_films, load_ratings, save_parquet, summarise  # noqa: E402
from filmscore.evaluate import rater_thin_data, ranking, reference_films, thin_data  # noqa: E402
from filmscore.scores import design, fit_prior  # noqa: E402

INK, MUTED, AXIS, SURFACE = "#0b0b0b", "#898781", "#c9c8c0", "#fcfcfb"
COLOURS = {"raw": "#eb6834", "imdb m=25": "#eda100", "pooled (flat)": "#1baf7a", "pooled (genre)": "#2a78d6"}
LABELS = {"raw": "Raw average", "imdb m=25": "IMDb formula, m = 25", "pooled (flat)": "Pooled, one overall average",
          "pooled (genre)": "Pooled, by genre and decade"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(axis="y", color="#e1e0d9", lw=0.8)
    ax.set_axisbelow(True)


def load(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = path.with_name(path.stem + "-ratings.parquet") if path.suffix == ".zip" else None
    if cache is not None and cache.exists():
        ratings = load_ratings(cache)
    else:
        ratings = load_ratings(path)
        if cache is not None:
            save_parquet(ratings, cache)
    return ratings, load_films(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "ml-32m.zip")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="fewer replicates, for a fast check")
    args = ap.parse_args()
    reps = 1 if args.quick else args.replicates
    args.out.mkdir(exist_ok=True)
    t0 = time.time()

    ratings, films = load(args.data)
    summary = summarise(ratings)
    ref = reference_films(summary, min_n=1000, folds=2, seed=0)
    print(f"{len(ratings):,} ratings, {len(summary):,} films, {len(ref):,} reference films")

    # Headline numbers about the data and the fitted model (all films)
    flat_all = fit_prior(summary)
    X_ref = design(films, ref.index, "genre")
    genre_ref = fit_prior(summary.loc[ref.index], X_ref, sigma2=flat_all.sigma2)
    flat_ref = fit_prior(summary.loc[ref.index], sigma2=flat_all.sigma2)
    info = {"ratings": int(len(ratings)), "films": int(len(summary)), "users": int(ratings["user"].nunique()),
            "reference_films": int(len(ref)), "median_ratings_per_film": float(summary["n"].median()),
            "share_films_5_or_fewer": float((summary["n"] <= 5).mean()),
            "share_films_1": float((summary["n"] == 1).mean()),
            "sigma": float(np.sqrt(flat_all.sigma2)),
            "all_films": {"tau_flat": flat_all.tau, "kappa_flat": flat_all.kappa},
            "reference": {"tau_flat": flat_ref.tau, "kappa_flat": flat_ref.kappa,
                          "tau_genre": genre_ref.tau, "kappa_genre": genre_ref.kappa}}

    # 1. Accuracy with n ratings
    ms = (1, 2, 3, 5, 10, 25, 50, 100, 250, 1000)
    thin = thin_data(ratings, films, summary, ref, ms=ms, replicates=reps)
    thin.to_csv(args.out / "thin_data.csv", index=False)
    rmse = thin.groupby(["n", "method"])["rmse"].mean().unstack("method")
    cov = thin.dropna(subset=["coverage"]).groupby(["n", "method"])["coverage"].mean().unstack("method")
    main_methods = ["raw", "imdb m=25", "pooled (flat)", "pooled (genre)", "pooled (genre), data-shaped prior"]
    t1 = rmse[main_methods].rename(columns=lambda c: LABELS.get(c, c.capitalize())).round(3)
    t2 = (100 * cov[["raw", "pooled (flat)", "pooled (genre)", "pooled (genre), data-shaped prior"]]).round(1)
    best_m = rmse[[f"imdb m={m}" for m in ms]].idxmin(axis=1).str.replace("imdb m=", "").astype(int)
    (args.out / "thin_data.md").write_text(
        f"Each of {len(ref):,} reference films shown n of its ratings at random; error against its full-data "
        f"average (stars). Mean over {reps} replicate(s) and 2 folds.\n\n## Root mean squared error\n\n"
        + t1.to_markdown() + "\n\n## Coverage of 90% intervals (%)\n\nThe raw average's interval is the "
        "usual one, average ± 1.645 sigma / sqrt(n).\n\n" + t2.to_markdown()
        + "\n\n## Best fixed m for the IMDb formula at each n\n\n" + best_m.to_frame("best m").to_markdown()
        + "\n")
    info["thin_rmse"] = {str(n): rmse.loc[n, main_methods].round(4).to_dict() for n in rmse.index}

    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    style(ax)
    for m in ["raw", "imdb m=25", "pooled (flat)", "pooled (genre)"]:
        ax.plot(rmse.index, rmse[m], marker="o", ms=5, lw=2, color=COLOURS[m], label=LABELS[m])
    ax.set_xscale("log")
    ax.set_xticks(rmse.index, [str(n) for n in rmse.index])
    ax.set_xlabel("Ratings shown per film", color=INK, fontsize=9.5)
    ax.set_ylabel("Typical error against the full-data average (stars)", color=INK, fontsize=9.5)
    ax.set_title("How far each score is from the truth, by number of ratings", loc="left", fontsize=11,
                 fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(args.out / "thin_data.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0), facecolor=SURFACE)
    style(ax)
    for n, colour in ((3, "#2a78d6"), (10, "#1baf7a"), (50, "#eb6834")):
        ax.plot(ms, [rmse.loc[n, f"imdb m={m}"] for m in ms], marker="o", ms=4, lw=2, color=colour,
                label=f"{n} ratings shown")
        ax.axhline(rmse.loc[n, "pooled (flat)"], color=colour, lw=1, ls="--")
    ax.axvline(info["reference"]["kappa_flat"], color=MUTED, lw=1)
    ax.annotate(f"estimated from the data: {info['reference']['kappa_flat']:.1f}",
                (info["reference"]["kappa_flat"], ax.get_ylim()[1]), xytext=(5, -12), textcoords="offset points",
                fontsize=8, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlabel("m in the IMDb formula (imaginary ratings at the overall average)", color=INK, fontsize=9.5)
    ax.set_ylabel("Typical error (stars)", color=INK, fontsize=9.5)
    ax.set_title("A fixed m must be guessed; the pooled model estimates it (dashed lines)", loc="left",
                 fontsize=11, fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(args.out / "fixed_m.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"thin data done ({time.time() - t0:.0f}s)")

    # 2. Ranking with a realistic spread of rating counts
    rank = ranking(ratings, films, summary, ref, count_pool=summary["n"].to_numpy(), tops=(20, 100),
                   replicates=4 * reps)
    rank.to_csv(args.out / "ranking.csv", index=False)
    cols = ["mean_shown", "mean_truth", "gap", "share_best10", "share_below_median", "worst_truth",
            "median_ratings_shown"]
    tab = rank.groupby(["top", "method"], sort=False)[cols].mean()
    tab[["share_best10", "share_below_median"]] *= 100
    names = {"mean_shown": "Score shown", "mean_truth": "True score", "gap": "Shown minus true",
             "share_best10": "Truly in best 10% (%)", "share_below_median": "Truly below median (%)",
             "worst_truth": "Worst true score", "median_ratings_shown": "Median ratings shown"}
    parts = []
    for top in (20, 100):
        parts.append(f"## Top {top}\n\n" + tab.loc[top].rename(columns=names).round(2).to_markdown())
    (args.out / "ranking.md").write_text(
        f"Each reference film shown a number of ratings drawn from the spread across all MovieLens films "
        f"(median {info['median_ratings_per_film']:.0f}), then ranked. Averages over {4 * reps} replicates.\n\n"
        + "\n\n".join(parts) + "\n")
    info["ranking_top20"] = tab.loc[20].round(4).to_dict(orient="index")
    print(f"ranking done ({time.time() - t0:.0f}s)")

    # 3. Allowing for who rated each film
    raters, rinfo = rater_thin_data(ratings, films, summary, ref, replicates=max(1, reps // 2))
    raters.to_csv(args.out / "raters.csv", index=False)
    rr = raters.groupby(["n", "method"])["rmse"].mean().unstack("method").round(3)
    (args.out / "raters.md").write_text(
        "Target: each reference film's rater-adjusted average (every rating minus its rater's habit, over "
        "all its ratings). Rater habits for the shown ratings come from a model fitted without the test "
        "films.\n\n## Root mean squared error (stars)\n\n" + rr.to_markdown() + "\n\n## Fitted model\n\n"
        + "\n".join(f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in rinfo.items())
        + "\n")
    info["raters"] = rinfo
    info["raters_rmse"] = {str(n): rr.loc[n].to_dict() for n in rr.index}
    info["seconds"] = round(time.time() - t0)
    (args.out / "summary.json").write_text(json.dumps(info, indent=2))
    print(f"all done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
