# Tutorials

All commands assume the current working directory is the repository root and that dependencies from `requirements.txt` are installed.

## Tutorial 1 — Reproduce the Jiuzhaigou temporal probability curve

This is the **minimum reproducible path** for reviewers. It uses the shipped sample features and pretrained weights; no catalog processing or training is required.

```bash
# Recommended one-command entry (Jiuzhaigou by default)
python main.py

# Other case studies
python main.py --event wenchuan
python main.py --event lushan
python main.py --event lushan2022
python main.py --event luding
python main.py --event jishishan
# or every event:
python main.py --event all
```

Or run the underlying scripts explicitly:

```bash
# 1) Ensemble inference
python scripts/predict.py \
  --input data/samples/test_jiuzhaigou.csv \
  --model-dir models \
  --output-dir outputs

# 2) Probability curve figure
python scripts/plot_probability.py \
  --input outputs/window_statistics_cs.csv \
  --output outputs/probability_curve.png
```

**What you should see**

- Console messages listing saved CSV/PNG paths under `outputs/`
- `outputs/probability_curve.png` with a red mean curve, shaded min–max band, and a vertical dashed line at day 0

**Optional**: skip the contribution multi-panel figure during inference:

```bash
python scripts/predict.py --no-contribution-plot
```

---

## Tutorial 2 — Spatially varying prior (SVP) map

```bash
python scripts/plot_prior_map.py \
  --catalog data/catalogs/2000-2013.csv \
  --start-date 2000-01-01 \
  --end-date 2014-12-31 \
  --output outputs/svp_spatial_map.png \
  --csv-output outputs/svp_grid.csv
```

**What you should see**

- A YlOrRd grid map of prior probabilities with numeric labels per cell
- `outputs/svp_grid.csv` with columns `lon_center`, `lat_center`, `earthquake_count`, `prior_prob`

Adjust `--lon-min/max`, `--lat-min/max`, and `--grid-size` to match your study region.

---

## Tutorial 3 — Spatial heatmap demo with sample grids

The repository includes eight example grid-point CSVs under `results/samples/` (near the Jiuzhaigou epicenter plus corner points). They are sufficient to exercise interpolation and plotting, not to reproduce a full publication map.

```bash
python scripts/plot_space_heatmaps.py \
  --data-dir results/samples \
  --output-dir outputs/heatmaps \
  --hypocenter-lon 103.82 \
  --hypocenter-lat 33.2 \
  --max-days 5
```

**What you should see**

- PNG files such as `outputs/heatmaps/heatmap_day_XXX.png`
- Active-fault polylines from `data/faults/active_faults_study_region.shp` overlaid by default (requires `cartopy` and `geopandas`)
- If `cartopy` is installed, a geographic axes is used; otherwise a plain Matplotlib axes is used

Disable the fault overlay:

```bash
python scripts/plot_space_heatmaps.py --no-faults --max-days 5
```

Or point to another shapefile:

```bash
python scripts/plot_space_heatmaps.py \
  --fault-shapefile /path/to/faults.shp \
  --max-days 5
```

### How to build a full spatial grid yourself

1. Generate test features at each grid node with `prepare_test_features.py` (or your own pipeline).
2. Run `predict.py` for each node and retain the mean probability series.
3. Save files named `statistics_lat_{lat}_lon_{lon}.csv` with columns  
   `Window_Index,Min_Probability,Max_Probability,Avg_Probability`  
   (percent scale 0–100 is expected by the heatmap colorbar).
4. Point `plot_space_heatmaps.py --data-dir` at that directory and set `--max-days 0` to plot all days.

---

## Tutorial 4 — HN cosine similarity (Wenchuan + Jiuzhaigou)

This reproduces the EQ vs NEQ cosine-similarity heatmap from LSTM hidden features.

```bash
python scripts/predict.py --event wenchuan --no-contribution-plot
python scripts/predict.py --event jiuzhaigou --no-contribution-plot
python scripts/plot_hn_cosine.py --events wenchuan jiuzhaigou --n-samples 100
```

**What you should see**

- `outputs/hn_cosine_wenchuan_jiuzhaigou.png`
- A block-structured heatmap with NEQ and EQ regions separated by dashed lines

If you already have a merged CSV with a `y_true` column:

```bash
python scripts/plot_hn_cosine.py --input path/to/hn_merged.csv --label-column y_true
```

## Tutorial 5 — (Optional) Retrain from a catalog

```bash
# Feature engineering (can take a long time on large catalogs)
python scripts/prepare_train_features.py \
  --catalog data/catalogs/2000-2013.csv \
  --output-dir data/processed \
  --random-nodes 10

# Training (GPU recommended)
python scripts/train.py \
  --train-dir data/processed \
  --model-number 5 \
  --output-dir models \
  --epochs 500
```

After training, repeat Tutorial 1 with the new weights.
