# Methodology

## 1. The problem

Most sites score a film by the average of its ratings. With few ratings that average is mostly luck. In
MovieLens 32M (32,000,204 ratings of 84,432 films by 200,948 people), half of all films have five ratings
or fewer and 22% have exactly one. Sorted by average rating, the top of the list is entirely films rated
once, by someone who gave them five stars.

The usual fixes are a minimum number of ratings (Rotten Tomatoes needs a set number of reviews before some
of its labels apply) or a formula with a fixed number of "imaginary" average ratings (IMDb's Top 250 once
published one). This project replaces both with the partial pooling used for surgical case durations in the
[theatre scheduler](https://github.com/jackcormacwright1/theatre_scheduler): estimate from all films how much
to trust a few ratings, then use each film's uncertainty when ranking. It also adjusts for who rated a film,
the equivalent of the theatre scheduler's surgeon effect, and it tests every method against a known answer,
as the [bed model](https://github.com/jackcormacwright1/bed-demand-model) does.

## 2. The model

### Notation

| Symbol | Meaning |
|---|---|
| *i* | a film |
| *n*ᵢ | the number of ratings film *i* has |
| *ȳ*ᵢ | the average of those ratings (the raw score) |
| *θ*ᵢ | the film's true score: the average it would settle at with unlimited ratings |
| *σ* | how far a single rating of a film typically sits from that film's true score |
| *x*ᵢ | the film's predictors: genres, release decade, and (for the whole catalogue) its number of ratings |
| *γ* | how much each predictor shifts a film's expected score |
| *xᵢ′γ* | the film's prediction: what its predictors alone suggest, before its own ratings |
| *τ* | how far true scores spread around their predictions |

The model has two levels:

* **A film's average is its true score plus noise.** *ȳ*ᵢ ~ Normal(*θ*ᵢ, *σ*²/*n*ᵢ). The noise shrinks as
  ratings accumulate: its standard deviation is *σ*/√*n*ᵢ.
* **True scores vary around their predictions.** *θ*ᵢ ~ Normal(*xᵢ′γ*, *τ*²).

### Each film's score

Given *σ*, *τ* and *γ*, the film's posterior (what we believe about *θ*ᵢ after seeing its ratings) is normal
with

* weight on its own ratings *w*ᵢ = *n*ᵢ / (*n*ᵢ + *κ*), where *κ* = *σ*²/*τ*²
* **pooled score** = *w*ᵢ × *ȳ*ᵢ + (1 − *w*ᵢ) × *xᵢ′γ*
* posterior standard deviation = √(*w*ᵢ *σ*²/*n*ᵢ)

*κ* acts as a number of imaginary ratings at the film's predicted value: a film with *κ* real ratings is
weighted half on its own average and half on its prediction.

**Worked example.** Take *σ* = 0.95 and *τ* = 0.38, close to the values fitted for well-known films with
genre and decade as predictors, so *κ* = 0.95² / 0.38² = 0.9025 / 0.1444 = 6.25. Film A has three ratings,
5, 4.5 and 5, so *ȳ* = 4.83, and its genre and decade predict 3.6. Film B has 300 ratings averaging 4.2 and
the same prediction.

| | Film A | Film B |
|---|---|---|
| Weight on own ratings | 3 / (3 + 6.25) = 0.324 | 300 / (300 + 6.25) = 0.980 |
| Pooled score | 0.324 × 4.83 + 0.676 × 3.6 = **4.00** | 0.980 × 4.2 + 0.020 × 3.6 = **4.19** |
| Posterior sd | √(0.324 × 0.9025 / 3) = 0.312 | √(0.980 × 0.9025 / 300) = 0.054 |
| 90% interval | 4.00 ± 1.645 × 0.312 = 3.49 to 4.51 | 4.10 to 4.28 |
| Lower bound (10th percentile) | 4.00 − 1.2816 × 0.312 = **3.60** | **4.12** |

On raw averages A beats B by 0.63 stars. Pooled, B is slightly ahead, and on the lower bound clearly ahead.

### Estimating *σ*, *τ* and *γ*

* *σ*² is the variance of ratings within a film, pooled over all films with at least two ratings and weighted
  by their degrees of freedom: Σ(*n*ᵢ − 1)*s*ᵢ² / Σ(*n*ᵢ − 1). In MovieLens, *σ* = 0.96 stars.
* *γ* and *τ* are estimated together from all films' averages, as in the theatre scheduler's grid model
  (Gelman et al., *Bayesian Data Analysis*, section 5.4). With a flat prior on *γ* and a uniform prior on
  *τ*, the posterior of *τ* is computed on a grid of 400 values from 0.005 to 2 stars:

  log p(*τ* | data) = −½ log|*X′WX*| − ½ Σ log *V*ᵢ − ½ Σ (*ȳ*ᵢ − *xᵢ′γ̂*)² / *V*ᵢ + constant,

  where *V*ᵢ = *τ*² + *σ*²/*n*ᵢ is the variance of film *i*'s average around its prediction, *W* is the
  diagonal matrix of 1/*V*ᵢ, and *γ̂* = (*X′WX*)⁻¹*X′Wȳ* is the weighted least-squares estimate at that *τ*.
* Each film's posterior is averaged over the grid posterior of *τ*, and includes the uncertainty in *γ̂*.
  With tens of thousands of films, *τ* is pinned down almost exactly (for all films, 0.500, 90% interval
  0.500 to 0.505).

### The IMDb formula is a special case

IMDb's published formula was weighted rating = *n*/(*n* + *m*) × *R* + *m*/(*n* + *m*) × *C*, where *R* is the
film's average, *C* the average across films and *m* a fixed number of votes. That is the pooled score with
no predictors and *m* in place of *κ*. The difference is that *m* has to be chosen, while *κ* is estimated.

## 3. What a film is pulled towards

Three versions of the prediction *xᵢ′γ*:

* **Flat:** one overall average for every film.
* **Genre and decade:** an intercept, one indicator per genre (a film can have several) and one per release
  decade, with the commonest decade as the baseline. Predictors that are combinations of others are dropped
  automatically.
* **Catalogue:** genre and decade plus bands of the film's number of ratings (1, 2, 3–5, 6–10, 11–25,
  26–100, 101–1,000, over 1,000). Films few people rate tend to be rated lower: films with one rating average
  2.87, films with over 1,000 average 3.40. So the number of ratings is itself evidence, and the dashboard
  uses this version for the whole catalogue. The tests in section 6 cannot use it, because there a popular
  film is deliberately shown only a few ratings.

## 4. Ranking: the lower bound

The pooled score is the best single estimate of a film's true score. Ranking is a different decision: from
thousands of films, pick the few that are best. Picking the highest estimates favours films whose estimates
are high partly by luck, so the chosen films disappoint on average (the "winner's curse"). The theatre
scheduler faced the same choice and booked lists so that the chance of overrunning stayed below 10%, rather
than booking to the average duration.

The equivalent here is to rank by the **lower bound**: the 10th percentile of the film's posterior, the score
we are 90% sure the film reaches. A film with few ratings has a wide posterior and so a low lower bound unless
its ratings are overwhelming; a film with many good ratings has a lower bound close to its score. Section 7.4
shows this picks better films than the pooled score itself.

## 5. Allowing for who rated a film

Some people rate everything generously and others harshly. The rater model is

*r*ᵤᵢ = *μ* + *a*ᵢ + *b*ᵤ + *e*ᵤᵢ, with *a*ᵢ ~ Normal(0, *τ*_film²), *b*ᵤ ~ Normal(0, *τ*_user²) and
*e*ᵤᵢ ~ Normal(0, *σ*_e²),

where *r*ᵤᵢ is user *u*'s rating of film *i*, *a*ᵢ the film's effect and *b*ᵤ the user's habit (positive for
someone who rates everything high). It is the surgeon effect of the theatre scheduler, and the "baseline
predictor" of recommender systems (Koren, 2008).

It is fitted by alternating: given the user effects, each film effect is its residuals' average shrunk
towards zero; given the film effects, each user effect likewise. After each pass both sets of effects are
re-centred on zero, with their averages moved into *μ*. At the solution they average exactly zero, and
without this step the alternating updates took hundreds of passes to settle the overall level; with it they
converge in 10 to 20. The three variances are re-estimated after each pass by an approximate EM step.

In MovieLens, *τ*_user = 0.44 stars: users' habits vary almost as much as films do (*τ*_film = 0.48). Taking
them out leaves single ratings less noisy (*σ*_e = 0.85 against 0.96), so a few ratings say more. A film's
rater-adjusted ratings are *r*ᵤᵢ − *b*ᵤ, and its rater-adjusted score applies section 2 to those.

## 6. Testing against a known answer

The 4,397 films with at least 1,000 ratings are the **reference films**. Their full-data average is known to
within about 0.03 stars, so it serves as the truth. They are split at random into two halves. For each half:

1. Everything a method learns from other films (*σ*, *τ*, *γ*, the IMDb formula's overall average *C*, the
   rater effects) is estimated from the **other** half only.
2. Each film is shown a random subset of its own ratings. The subsets are nested: the five ratings a film is
   shown at *n* = 5 are among the ten it is shown at *n* = 10.
3. Each method's estimate from the subset is compared with the film's truth.

Two experiments:

* **Thin data:** every film is shown the same *n* ratings, for *n* from 1 to 100; five replicates with
  different subsets.
* **Ranking:** every film is shown a number of ratings drawn at random from the real spread of rating
  counts across all 84,432 films (median 5), then films are ranked; 20 replicates. What are the films each
  method puts at the top really like?

A third, for the rater model, changes the target to the rater-adjusted average (every rating minus its
rater's habit, over all the film's ratings), with rater effects for the shown ratings estimated without any
of the test films' ratings.

These reference films are popular films pretending to be obscure. That matches a film early in its life,
which will eventually have many ratings; it does not test films that will always be obscure, whose truth
cannot be known.

## 7. Results

### 7.1 Accuracy with few ratings

Root mean squared error against the truth (stars):

| Ratings shown | Raw average | IMDb formula, *m* = 25 | Pooled, flat | Pooled, genre and decade |
|---:|---:|---:|---:|---:|
| 1 | 0.966 | 0.453 | 0.424 | **0.361** |
| 2 | 0.684 | 0.438 | 0.389 | **0.339** |
| 3 | 0.556 | 0.424 | 0.359 | **0.319** |
| 5 | 0.432 | 0.398 | 0.318 | **0.289** |
| 10 | 0.307 | 0.347 | 0.257 | **0.241** |
| 20 | 0.217 | 0.278 | 0.197 | **0.189** |
| 50 | 0.136 | 0.181 | 0.130 | **0.128** |
| 100 | 0.095 | 0.121 | 0.094 | **0.093** |

With one rating the pooled score's typical error is 0.36 stars against 0.97 for the raw average. Its
advantage fades as ratings accumulate, as it should: by 100 ratings all the scores agree. Genre and decade
help most when a film has very few ratings.

### 7.2 The IMDb formula needs the right *m*

The best fixed *m* was 5 at every *n* tested. The pooled model estimates the equivalent, *κ*, from the data:
4.2 with a flat prior, matching the best fixed choice without any tuning. A conventional-looking *m* = 25 is
worse than the raw average once a film has somewhere between five and ten ratings, because it keeps pulling well-rated films
towards the middle long after their own ratings are reliable (`results/fixed_m.png`).

### 7.3 The intervals are honest

90% intervals from the pooled model contained the truth 89.6% to 90.8% of the time at every *n*. The raw
average's usual interval (average ± 1.645*σ*/√*n*) is also about right, but with one rating it is about
±1.6 stars wide, so it says little.

### 7.4 Picking the top 20

Each film shown a realistic number of ratings, then ranked. Averages over 20 replicates:

| Ranked by | Score shown | True score | Shown minus true | Truly in best 10% | Worst true score | Median ratings shown |
|---|---:|---:|---:|---:|---:|---:|
| Raw average | 5.00 | 3.66 | +1.34 | 26% | 2.72 | 1 |
| Raw average, 20+ ratings | 4.27 | 4.13 | +0.14 | 93% | 3.85 | 50 |
| IMDb formula, *m* = 25 | 4.12 | 4.15 | −0.03 | 100% | 4.05 | 842 |
| Pooled, flat | 4.23 | 4.14 | +0.09 | 93% | 3.83 | 41 |
| Pooled, genre and decade | 4.31 | 4.10 | +0.21 | 83% | 3.74 | 8 |
| Lower bound, flat | 4.11 | 4.16 | −0.04 | 99.5% | 4.03 | 639 |
| Lower bound, genre and decade | 4.12 | 4.16 | −0.04 | 97% | 3.97 | 530 |

* **The raw average fails completely.** Every one of its top 20 has a single five-star rating, and a quarter
  of them are truly below the median film.
* **The best estimate is not the best ranking.** The pooled score is the most accurate estimate for a
  typical film (7.1), but the films it ranks highest are those whose few ratings happened to be strongest,
  so they are overrated by 0.09 stars (flat) to 0.21 stars (genre and decade). The genre version does worse
  here because it starts some genres (documentaries, animation) high, and then promotes members of those
  genres with a handful of good ratings. Its predictions are not badly wrong on average (within about 0.07
  stars of the truth in every tenth of predictions); the problem is what ranking selects for.
* **The lower bound fixes it.** Its top 20 are the best films of any method (true score 4.16) and it
  promises slightly less than it delivers (−0.04), whichever prediction it uses.
* **A large fixed *m* works for the top 20, but not further down.** The IMDb formula with *m* = 25 matches
  the lower bound for the top 20, because a large *m* strongly favours films with many ratings. For the top
  100, 76% of its picks are truly in the best 10%, against 86% for the lower bound with genre and decade;
  with *m* = 100 it falls to 58%. The lower bound adjusts each film's penalty to that film's uncertainty,
  rather than applying one penalty to all.

The best *m* for estimating a film's score (about 5) and for ranking (larger) differ, because they are
different decisions. The lower bound handles the second without a second tuning constant.

### 7.5 A check: does the bell-curve prior matter?

True film scores are skewed: a long tail of bad films and a short tail of great ones (skewness −0.58 around
the genre and decade predictions). To check whether the normal prior's shape distorts the scores, the
prediction's spread was also estimated without assuming a shape: the nonparametric maximum likelihood
estimate of the mixing distribution on a grid of 300 points (Kiefer and Wolfowitz, 1956), by the EM
algorithm of Laird (1978). It changed the error by at most 0.002 stars at every *n*, and its 90% intervals
covered 89.1% to 89.4%. The bell-curve assumption is not what limits the scores.

### 7.6 Allowing for who rated

Here the target is the rater-adjusted average: how an average rater would score the film. Root mean squared
error (stars):

| Ratings shown | Raw average | Pooled, genre and decade | Pooled, rater-adjusted |
|---:|---:|---:|---:|
| 1 | 0.984 | 0.383 | **0.354** |
| 3 | 0.575 | 0.347 | **0.307** |
| 5 | 0.453 | 0.318 | **0.274** |
| 10 | 0.334 | 0.275 | **0.223** |
| 20 | 0.260 | 0.235 | **0.173** |
| 50 | 0.198 | 0.191 | **0.117** |

Taking out each rater's habit before pooling cuts the error by 14% with five ratings and by 39% with fifty.
Part of the gain at larger *n* comes from the target itself: a method that ignores raters estimates the
average of the film's actual raters, habits included, and for these popular films the two differ with a
standard deviation of 0.06 stars across films. For films with
thousands of ratings the two agree closely (correlation 0.992), because generous and harsh raters cancel
out; for a film with five ratings they need not.

## 8. Limitations

* **MovieLens users are not critics.** They are self-selected users of a film recommendation site, rating
  films they chose to watch. The method applies to critic reviews, but these numbers do not.
* **Who watches what is not random.** People choose films they expect to like, so a film's raters are not a
  random sample of everyone. The scores estimate what the film's eventual raters would give it; rater effects
  adjust for generosity, not for taste.
* **Ratings are not normal.** They are half-star steps between 0.5 and 5. The normal model is an
  approximation; the calibration checks (7.3) suggest it is a good one for averages.
* **One *σ* for every film.** Films differ in how much their ratings spread (polarising films more), and
  spread falls as quality rises (correlation −0.65 across reference films). A film-specific *σ* would be a
  natural extension.
* **The tests use popular films.** Their truth is known only because they are popular. Films that will
  always have few ratings may differ in ways the tests cannot show.
* **Ratings span 1995 to 2023.** Tastes and the user base change over time; the model ignores that.

## 9. References

* Gelman A, Carlin JB, Stern HS, Dunson DB, Vehtari A, Rubin DB (2013). *Bayesian Data Analysis*, 3rd edn.
  CRC Press. Section 5.4.
* Efron B, Morris C (1975). Data analysis using Stein's estimator and its generalizations. *Journal of the
  American Statistical Association* 70(350), 311–319.
* Harper FM, Konstan JA (2015). The MovieLens datasets: history and context. *ACM Transactions on
  Interactive Intelligent Systems* 5(4), 19. https://doi.org/10.1145/2827872
* Kiefer J, Wolfowitz J (1956). Consistency of the maximum likelihood estimator in the presence of infinitely
  many incidental parameters. *Annals of Mathematical Statistics* 27(4), 887–906.
* Koren Y (2008). Factorization meets the neighborhood: a multifaceted collaborative filtering model.
  *Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*,
  426–434.
* Laird N (1978). Nonparametric maximum likelihood estimation of a mixing distribution. *Journal of the
  American Statistical Association* 73(364), 805–811.
* Miller E (2009). How not to sort by average rating. https://www.evanmiller.org/how-not-to-sort-by-average-rating.html
