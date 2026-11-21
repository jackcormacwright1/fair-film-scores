"""Fair film scores: a Streamlit dashboard over MovieLens 32M.

Run locally:   streamlit run app.py
The per-film table it reads (app_data/films.parquet) is built by scripts/build_app_data.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(page_title="Fair film scores", page_icon="🎬", layout="wide")

APP = ROOT / "app_data"
RESULTS = ROOT / "results"
METHODOLOGY_FILE = ROOT / "docs" / "METHODOLOGY.md"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
NEUTRAL, MUTED, GRID, BAND = "#52514e", "#898781", "#e1e0d9", "#f0efea"

st.markdown(f"""
<style>
.block-container {{ padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1320px; }}
.eyebrow {{ color: {NEUTRAL}; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
            margin: 0.4rem 0 0.35rem; }}
.lede {{ color: #3a3936; font-size: 1.02rem; margin: -0.4rem 0 0.6rem; max-width: 62rem; }}
.kpi {{ background: #ffffff; border: 1px solid {GRID}; border-radius: 10px; padding: 0.8rem 1rem 0.85rem; height: 100%; }}
.kpi .label {{ color: {NEUTRAL}; font-size: 0.86rem; line-height: 1.3; min-height: 2.2rem; }}
.kpi .value {{ color: #1d1d1b; font-size: 2rem; font-weight: 600; line-height: 1.15; margin: 0.15rem 0 0.3rem;
               font-variant-numeric: tabular-nums; white-space: nowrap; }}
.kpi .note {{ color: {MUTED}; font-size: 0.78rem; line-height: 1.35; }}
@media (max-width: 1100px) {{ .kpi .value {{ font-size: 1.45rem; }} }}
.callout {{ background: #ffffff; border: 1px solid {GRID}; border-left: 4px solid {BLUE}; border-radius: 8px;
            padding: 0.85rem 1.1rem; margin: 0.3rem 0 1rem; }}
.sum {{ font-variant-numeric: tabular-nums; background: {BAND}; border-radius: 8px; padding: 0.7rem 1rem;
        margin: 0.4rem 0 0.8rem; line-height: 1.7; }}
.chart-note {{ color: {MUTED}; font-size: 0.8rem; margin: -0.3rem 0 0.4rem; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading 84,000 films…")
def load_films() -> pd.DataFrame:
    films = pd.read_parquet(APP / "films.parquet")
    films["label"] = films["title"] + films["year"].map(lambda y: "" if pd.isna(y) else f" ({int(y)})")
    films["decade"] = (films["year"] // 10 * 10).astype("Float32")
    return films


@st.cache_data
def load_model() -> dict:
    return json.loads((APP / "model.json").read_text())


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    path = RESULTS / name
    return pd.read_csv(path) if path.exists() else None


if not (APP / "films.parquet").exists():
    st.error("app_data/films.parquet is missing: run scripts/build_app_data.py first.")
    st.stop()

films = load_films()
model = load_model()
KAPPA = model["kappa"]
LOWER = 100 - round(100 * model["lower_q"])
GENRES = sorted({g for gs in films["genres"] for g in gs.split("|") if g})


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="chart-note">{text}</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, text: str) -> str:
    return (f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
            f'<div class="note">{text}</div></div>')


def stars(x: float) -> str:
    return "–" if pd.isna(x) else f"{x:.2f}"


# ---------------------------------------------------------------- header
eyebrow("MovieLens 32M · 84,432 films · 32 million ratings")
st.title("Fair film scores")
st.markdown('<p class="lede">A film rated five stars by one person beats every classic on a plain average. '
            "These scores allow for how many ratings each film has, and whose, using the same partial pooling "
            "as the theatre scheduler's case durations.</p>", unsafe_allow_html=True)

n = films["n"]
cols = st.columns(4)
cards = [
    kpi("Ratings per film, median", f"{n.median():.0f}", "Half of all films have five ratings or fewer."),
    kpi("Films with one rating", f"{(n == 1).mean():.0%}", "Each one's raw average is a single person's opinion."),
    kpi("Weight of the prediction", f"{KAPPA:.1f} ratings",
        "Before its own ratings count for much, a film's score leans on what its genre, decade and number "
        "of ratings predict, as if that were worth this many ratings."),
    kpi("Spread of single ratings", f"± {model['sigma']:.2f} stars",
        "How far one person's rating of a film typically sits from the film's average."),
]
for c, card in zip(cols, cards):
    c.markdown(card, unsafe_allow_html=True)
st.write("")

tab_top, tab_film, tab_test, tab_about = st.tabs(["Top films", "Look up a film", "Does it work?", "Methodology"])

# ---------------------------------------------------------------- top films
with tab_top:
    METHODS = {
        "raw": "Raw average",
        "raw_min": "Raw average, with a minimum number of ratings",
        "imdb": "IMDb formula (fixed m)",
        "score": "Pooled score (best estimate)",
        "lower_bound": f"Lower bound ({LOWER}% sure it is at least this good)",
    }
    left, right = st.columns([1, 2.4], gap="large")
    with left:
        eyebrow("Rank by")
        method = st.radio("Rank by", list(METHODS), format_func=METHODS.get, index=4, label_visibility="collapsed")
        min_n = 1
        if method == "raw_min":
            min_n = st.select_slider("Minimum ratings", [2, 5, 10, 20, 50, 100, 500, 1000], value=100)
        m = KAPPA
        if method == "imdb":
            m = st.select_slider("m: imaginary ratings at the overall average",
                                 [1, 2, 5, 10, 25, 50, 100, 250, 1000, 10000], value=25)
            st.caption(f"The pooled model estimates its own equivalent from the data: about {KAPPA:.1f}.")
        adjusted = False
        if method in ("score", "lower_bound"):
            adjusted = st.toggle("Allow for who rated each film", value=False,
                                 help="Takes out each rater's habit of scoring high or low before averaging.")
            if adjusted:
                st.caption("Adjusted scores are on the scale of a typical user rather than a typical rating. The "
                           "people who rate most films are harsher than average, so adjusted scores run about "
                           "0.1 stars higher.")
        eyebrow("Filter")
        genres = st.multiselect("Genres (any of)", GENRES, placeholder="All genres")
        years = st.slider("Released", 1900, 2023, (1900, 2023))
        top_k = st.select_slider("Films to show", [10, 20, 50, 100], value=20)

    d = films
    if genres:
        d = d[d["genres"].map(lambda g: any(x in g.split("|") for x in genres))]
    d = d[(d["year"].isna() & (years == (1900, 2023))) | d["year"].between(years[0], years[1])]
    d = d[d["n"] >= min_n]
    if method in ("raw", "raw_min"):
        key = d["raw"]
    elif method == "imdb":
        overall = model["film_average_mean"]
        key = (d["n"] * d["raw"] + m * overall) / (d["n"] + m)
    else:
        key = d[("adj_" if adjusted else "") + method]
    d = d.assign(key=key).sort_values(["key", "n"], ascending=[False, False]).head(top_k)
    pre = "adj_" if adjusted else ""

    with right:
        shown = d[["label", "genres", "n", "raw", f"{pre}score", f"{pre}low", f"{pre}high"]].copy()
        shown.insert(0, "Rank", np.arange(1, len(shown) + 1))
        one = (d["n"] == 1).sum()
        med = d["n"].median() if len(d) else 0
        msg = (f"The top {len(d)} have a median of <b>{med:,.0f}</b> ratings each"
               + (f", and <b>{one}</b> of them have just one." if one else "."))
        if method == "raw":
            msg += " A plain average is topped by whoever happened to give a single five-star rating."
        elif method == "raw_min":
            msg += f" A minimum of {min_n} hides the flukes, but also every film with fewer ratings, good or bad."
        elif method == "imdb":
            msg += f" With m = {m:g}, every film starts as if it had {m:g} ratings at the overall average."
        elif method == "score":
            msg += " Films with few ratings can still rank highly, but only if their ratings are very strong."
        else:
            msg += (" Ranking by the lower bound penalises uncertainty, so films need either many good "
                    "ratings or overwhelming few.")
        st.markdown(f'<div class="callout">{msg}</div>', unsafe_allow_html=True)

        chart_d = d.assign(rank=np.arange(1, len(d) + 1), score=d[f"{pre}score"], low=d[f"{pre}low"],
                           high=d[f"{pre}high"], bound=d[f"{pre}lower_bound"])
        lo_x = float(np.floor(min(chart_d[["low", "raw", "bound"]].min().min(), 4.0) * 4) / 4) if len(d) else 2.5
        hi_x = float(min(5.05, np.ceil(chart_d[["high", "raw"]].max().max() * 10) / 10 + 0.05)) if len(d) else 5.05
        base = alt.Chart(chart_d).encode(y=alt.Y("label:N", sort=None, title=None,
                                                 axis=alt.Axis(labelLimit=320, labelColor=NEUTRAL)))
        rule = base.mark_rule(color=BLUE, strokeWidth=2, opacity=0.5).encode(
            x=alt.X("low:Q", title="Stars", scale=alt.Scale(domain=[lo_x, hi_x])), x2="high:Q")
        tick = base.mark_tick(color="#1d1d1b", thickness=2, size=12).encode(x="bound:Q")
        point = base.mark_circle(color=BLUE, size=70, opacity=1).encode(
            x="score:Q", tooltip=[alt.Tooltip("label:N", title="Film"), alt.Tooltip("n:Q", title="Ratings", format=","),
                                  alt.Tooltip("raw:Q", title="Raw average", format=".2f"),
                                  alt.Tooltip("score:Q", title="Pooled score", format=".2f"),
                                  alt.Tooltip("bound:Q", title="Lower bound", format=".2f"),
                                  alt.Tooltip("low:Q", title="90% interval from", format=".2f"),
                                  alt.Tooltip("high:Q", title="to", format=".2f")])
        rawpt = base.mark_point(color=ORANGE, size=60, filled=False, strokeWidth=1.6).encode(x="raw:Q")
        st.altair_chart((rule + tick + point + rawpt).properties(height=max(240, 24 * len(d))),
                        use_container_width=True)
        note(f"<span style='color:{BLUE}'>●</span> pooled score with its 90% interval &nbsp; "
             f"<b>|</b> lower bound &nbsp; <span style='color:{ORANGE}'>○</span> raw average")

        shown = shown.rename(columns={"label": "Film", "genres": "Genres", "n": "Ratings", "raw": "Raw average",
                                      f"{pre}score": "Pooled score", f"{pre}low": "90% from", f"{pre}high": "to"})
        shown["Genres"] = shown["Genres"].str.replace("|", ", ", regex=False)
        st.dataframe(shown, hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="%.2f")
                                    for c in ["Raw average", "Pooled score", "90% from", "to"]}
                     | {"Ratings": st.column_config.NumberColumn(format="%d")})

# ---------------------------------------------------------------- look up a film
with tab_film:
    eyebrow("Find a film")
    query = st.text_input("Title contains", value="Attenborough", label_visibility="collapsed",
                          placeholder="Type part of a title")
    matches = films[films["title"].str.contains(query, case=False, regex=False)] if query else films.head(0)
    matches = matches.sort_values("n", ascending=False).head(60)
    if matches.empty:
        st.info("No film matches that title.")
    else:
        default = matches.index.get_indexer(matches.index[matches["title"] == "David Attenborough's Tasmania"])
        pick = st.selectbox("Film", matches.index, index=int(default[0]) if len(default) else 0,
                            format_func=lambda i: f"{matches.loc[i, 'label']} · {matches.loc[i, 'n']:,} ratings",
                            label_visibility="collapsed")
        f = films.loc[pick]
        w = float(f["weight"])
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(kpi("Raw average", stars(f["raw"]), f"From {f['n']:,} rating{'s' if f['n'] != 1 else ''}."),
                    unsafe_allow_html=True)
        c2.markdown(kpi("Prediction", stars(f["prediction"]),
                        "What its genres, decade and number of ratings suggest before its own ratings count."),
                    unsafe_allow_html=True)
        c3.markdown(kpi("Pooled score", stars(f["score"]), f"90% interval {stars(f['low'])} to {stars(f['high'])}."),
                    unsafe_allow_html=True)
        c4.markdown(kpi(f"Lower bound", stars(f["lower_bound"]),
                        f"{LOWER}% sure the film's true score is at least this."), unsafe_allow_html=True)
        eyebrow("The arithmetic")
        st.markdown(
            f'<div class="sum">weight on its own ratings = {f["n"]:,} ÷ ({f["n"]:,} + {KAPPA:.1f}) = '
            f"<b>{w:.3f}</b><br>pooled score = {w:.3f} × {f['raw']:.2f} + (1 − {w:.3f}) × {f['prediction']:.2f} "
            f"= <b>{f['score']:.2f}</b></div>", unsafe_allow_html=True)
        st.caption(f"{KAPPA:.1f} is sigma² ÷ tau²: how noisy one rating is ({model['sigma']:.2f}²) over how much "
                   f"true scores vary beyond what the prediction explains ({model['tau']:.2f}²). The weight here "
                   "is averaged over the uncertainty in tau, so it can differ from the formula in the third "
                   "decimal place.")
        eyebrow("Allowing for who rated it")
        st.markdown(
            f"Taking out each rater's habit (some people rate everything high), its raw average becomes "
            f"**{stars(f['adj_raw'])}** and its pooled score **{stars(f['adj_score'])}** "
            f"(90% interval {stars(f['adj_low'])} to {stars(f['adj_high'])})."
            + (" An adjusted average can pass 5 stars: a five from someone who rates harshly counts for more."
               if f["adj_raw"] > 5 else ""))
        eyebrow("If its future ratings averaged the same")
        if f["n"] >= 1000:
            st.markdown(f"With {f['n']:,} ratings this film's score has settled: more ratings like these would "
                        "barely move it.")
        else:
            grid = np.unique(np.round(np.geomspace(f["n"], max(300, 30 * f["n"]), 40)).astype(int))
            sd1 = model["sigma"]
            tau2 = model["tau"] ** 2
            wt = tau2 / (tau2 + sd1 ** 2 / grid)
            proj = pd.DataFrame({"ratings": grid, "score": wt * f["raw"] + (1 - wt) * f["prediction"],
                                 "sd": np.sqrt(wt * sd1 ** 2 / grid)})
            proj["low"], proj["high"] = proj["score"] - 1.645 * proj["sd"], proj["score"] + 1.645 * proj["sd"]
            y_lo = float(np.floor(min(proj["low"].min(), f["raw"], f["prediction"]) * 4) / 4)
            y_hi = float(min(5.25, np.ceil(max(proj["high"].max(), f["raw"]) * 4) / 4))
            ys = alt.Scale(domain=[y_lo, y_hi])
            b = alt.Chart(proj).encode(x=alt.X("ratings:Q", title="Number of ratings", scale=alt.Scale(
                type="log", domain=[int(grid[0]), int(grid[-1])], nice=False)))
            band = b.mark_area(color=BLUE, opacity=0.15).encode(y=alt.Y("low:Q", title="Stars", scale=ys), y2="high:Q")
            line = b.mark_line(color=BLUE, strokeWidth=2).encode(
                y=alt.Y("score:Q", scale=ys), tooltip=[alt.Tooltip("ratings:Q", format=","),
                                                       alt.Tooltip("score:Q", format=".2f")])
            rawline = alt.Chart(pd.DataFrame({"y": [f["raw"]]})).mark_rule(color=ORANGE, strokeDash=[5, 4]).encode(
                y=alt.Y("y:Q", scale=ys))
            st.altair_chart((band + line + rawline).properties(height=280), use_container_width=True)
            note("If every new rating matched its current average (dashed), the pooled score would climb or fall "
                 "towards it and its 90% interval would narrow. Approximate: uses the overall tau and ignores "
                 "rater effects.")

# ---------------------------------------------------------------- does it work
with tab_test:
    thin = load_csv("thin_data.csv")
    rank = load_csv("ranking.csv")
    raters = load_csv("raters.csv")
    st.markdown("The test uses films with at least 1,000 ratings, whose full average is all but certain. Each "
                "is shown only a few of its ratings at random, and each method's estimate is compared with the "
                "full average. Everything a method learns from other films comes from a separate half of the "
                "films, so no film helps to judge itself.")
    if thin is not None:
        eyebrow("Accuracy with few ratings")
        names = {"raw": "Raw average", "imdb m=25": "IMDb formula, m = 25", "pooled (flat)": "Pooled, overall average",
                 "pooled (genre)": "Pooled, genre and decade"}
        t = (thin[thin["method"].isin(names)].groupby(["n", "method"], as_index=False)["rmse"].mean()
             .assign(method=lambda x: x["method"].map(names)))
        ch = alt.Chart(t).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("n:Q", scale=alt.Scale(type="log"), title="Ratings shown per film"),
            y=alt.Y("rmse:Q", title="Typical error (stars)"),
            color=alt.Color("method:N", title=None, scale=alt.Scale(domain=list(names.values()),
                                                                     range=[ORANGE, YELLOW, AQUA, BLUE]),
                            legend=alt.Legend(orient="top")),
            tooltip=["method", "n", alt.Tooltip("rmse:Q", format=".3f")])
        st.altair_chart(ch.properties(height=320), use_container_width=True)
        note("Root mean squared error against the full-data average, averaged over replicates and both halves.")
    if rank is not None:
        eyebrow("Picking the top 20")
        r = rank[rank["top"] == 20].groupby("method", sort=False)[
            ["mean_truth", "gap", "share_best10", "share_below_median", "median_ratings_shown"]].mean()
        r[["share_best10", "share_below_median"]] *= 100
        r["median_ratings_shown"] = r["median_ratings_shown"].round(0)
        r.index = (r.index.str.replace("raw average", "Raw average").str.replace("pooled (flat)", "Pooled, overall average", regex=False)
                   .str.replace("pooled (genre)", "Pooled, genre and decade", regex=False))
        r.index.name = "Ranked by"
        st.markdown("Each film shown a realistic number of ratings (drawn from the real spread: half of films "
                    "have five or fewer), then ranked. What are the 20 films each method puts first really like?")
        st.dataframe(r.rename(columns={"mean_truth": "True score of picks", "gap": "Shown minus true",
                                       "share_best10": "Truly in best 10% (%)",
                                       "share_below_median": "Truly below median (%)",
                                       "median_ratings_shown": "Median ratings shown"}).round(2),
                     use_container_width=True)
    if raters is not None:
        eyebrow("Allowing for who rated")
        rr = raters.groupby(["n", "method"])["rmse"].mean().unstack("method").round(3)
        rr = rr[["raw average", "pooled (genre)", "pooled (genre), rater-adjusted"]].rename(columns={
            "raw average": "Raw average", "pooled (genre)": "Pooled, genre and decade",
            "pooled (genre), rater-adjusted": "Pooled, rater-adjusted"})
        rr.index.name = "Ratings shown"
        st.markdown("Here the target is the rater-adjusted average: how an average rater would score the film. "
                    "Typical error in stars:")
        st.dataframe(rr, use_container_width=True)

# ---------------------------------------------------------------- methodology
with tab_about:
    st.markdown(METHODOLOGY_FILE.read_text(encoding="utf-8") if METHODOLOGY_FILE.exists()
                else "The methodology is in docs/METHODOLOGY.md in the project repository.")

st.caption("Data: MovieLens 32M, GroupLens Research, University of Minnesota (F. M. Harper and J. A. Konstan, 2015, "
           "The MovieLens Datasets: History and Context, ACM TiiS 5(4)). Used under the MovieLens licence; this "
           "project is not endorsed by the University of Minnesota or GroupLens.")
