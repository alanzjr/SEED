# Spatial result samples

These CSV files are a **small subset** of a denser regional grid used for publication figures. They exist so that `scripts/plot_space_heatmaps.py` can be demonstrated without shipping tens of megabytes of intermediate results.

Filename pattern: `statistics_lat_{lat}_lon_{lon}.csv`

Columns: `Window_Index`, `Min_Probability`, `Max_Probability`, `Avg_Probability`

See [docs/TUTORIAL.md](../../docs/TUTORIAL.md) Tutorial 3 for usage and instructions to regenerate a full grid.
