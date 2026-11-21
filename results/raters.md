Target: each reference film's rater-adjusted average (every rating minus its rater's habit, over all its ratings). Rater habits for the shown ratings come from a model fitted without the test films.

## Root mean squared error (stars)

|   n |   pooled (genre) |   pooled (genre), rater-adjusted |   raw average |
|----:|-----------------:|---------------------------------:|--------------:|
|   1 |            0.383 |                            0.354 |         0.984 |
|   2 |            0.365 |                            0.329 |         0.703 |
|   3 |            0.347 |                            0.307 |         0.575 |
|   5 |            0.318 |                            0.274 |         0.453 |
|  10 |            0.275 |                            0.223 |         0.334 |
|  20 |            0.235 |                            0.173 |         0.26  |
|  50 |            0.191 |                            0.117 |         0.198 |

## Fitted model

- tau_user: 0.439
- tau_film: 0.482
- sigma: 0.850
- iterations: 14
- corr_raw_vs_adjusted_truth: 0.992
- sd_raw_minus_adjusted_truth: 0.060
- iterations_fold0: 18
- sigma_raw_fold0: 0.950
- sigma_adjusted_fold0: 0.843
- iterations_fold1: 18
- sigma_raw_fold1: 0.956
- sigma_adjusted_fold1: 0.848
