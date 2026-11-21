Each reference film shown a number of ratings drawn from the spread across all MovieLens films (median 5), then ranked. Averages over 20 replicates.

## Top 20

| method                      |   Score shown |   True score |   Shown minus true |   Truly in best 10% (%) |   Truly below median (%) |   Worst true score |   Median ratings shown |
|:----------------------------|--------------:|-------------:|-------------------:|------------------------:|-------------------------:|-------------------:|-----------------------:|
| raw average                 |          5    |         3.66 |               1.34 |                   25.5  |                       27 |               2.72 |                   1    |
| raw average, 20+ ratings    |          4.27 |         4.13 |               0.14 |                   92.5  |                        0 |               3.85 |                  50.35 |
| IMDb formula, m = 25        |          4.12 |         4.15 |              -0.03 |                  100    |                        0 |               4.05 |                 841.65 |
| IMDb formula, m = 100       |          4.06 |         4.11 |              -0.05 |                  100    |                        0 |               4    |                1756.72 |
| pooled (flat)               |          4.23 |         4.14 |               0.09 |                   92.5  |                        0 |               3.83 |                  40.8  |
| pooled (flat), lower bound  |          4.11 |         4.16 |              -0.04 |                   99.5  |                        0 |               4.03 |                 638.92 |
| pooled (genre)              |          4.31 |         4.1  |               0.21 |                   82.75 |                        0 |               3.74 |                   7.8  |
| pooled (genre), lower bound |          4.12 |         4.16 |              -0.04 |                   97.25 |                        0 |               3.97 |                 530    |

## Top 100

| method                      |   Score shown |   True score |   Shown minus true |   Truly in best 10% (%) |   Truly below median (%) |   Worst true score |   Median ratings shown |
|:----------------------------|--------------:|-------------:|-------------------:|------------------------:|-------------------------:|-------------------:|-----------------------:|
| raw average                 |          5    |         3.67 |               1.33 |                   25.75 |                     24.4 |               2.41 |                   1    |
| raw average, 20+ ratings    |          4.11 |         4.04 |               0.07 |                   81.85 |                      0   |               3.68 |                  94.92 |
| IMDb formula, m = 25        |          3.98 |         4.03 |              -0.04 |                   75.95 |                      0   |               3.83 |                 456.08 |
| IMDb formula, m = 100       |          3.9  |         3.98 |              -0.08 |                   57.65 |                      0   |               3.79 |                 974.55 |
| pooled (flat)               |          4.11 |         4.05 |               0.06 |                   82.8  |                      0.4 |               3.45 |                  46.12 |
| pooled (flat), lower bound  |          3.98 |         4.04 |              -0.06 |                   81.85 |                      0   |               3.77 |                 335.15 |
| pooled (genre)              |          4.17 |         4.05 |               0.13 |                   79.7  |                      0.2 |               3.54 |                  10.55 |
| pooled (genre), lower bound |          3.99 |         4.05 |              -0.06 |                   85.65 |                      0   |               3.77 |                 260.62 |
