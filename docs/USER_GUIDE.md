# User Guide

This guide describes inputs, outputs, options, and expected behaviour for the SEED command-line tools. Run all commands from the **repository root**.

## `main.py`

Top-level entry point for reproducing the main results.

```bash
python main.py                         # demo for Jiuzhaigou
python main.py --event wenchuan        # demo for another event
python main.py --event all             # demo for every event
python main.py --mode prior            # SVP map
python main.py --mode space            # sample spatial heatmaps
python main.py --mode all              # demo + prior + space
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--mode` | `demo` | `demo` / `prior` / `space` / `all` |
| `--event` | `jiuzhaigou` | Case study, or `all` |
| `--no-contribution-plot` | off | Skip contribution figure in demo |
| `--max-days` | `5` | Days for spatial heatmaps (`0` = all) |

### Case-study events

| Event key | Test features | Model directory |
|-----------|---------------|-----------------|
| `jiuzhaigou` | `data/samples/test_jiuzhaigou.csv` | `models/jiuzhaigou/` |
| `wenchuan` | `data/samples/test_wenchuan.csv` | `models/wenchuan/` |
| `lushan` | `data/samples/test_lushan.csv` | `models/lushan/` |
| `lushan2022` | `data/samples/test_lushan2022.csv` | `models/lushan2022/` |
| `luding` | `data/samples/test_luding.csv` | `models/luding/` |
| `jishishan` | `data/samples/test_jishishan.csv` | `models/jishishan/` |

`predict.py` auto-discovers every `model_{i}_event_{j}.pth` + `RF_model_{i}_event_{j}.joblib` pair in the event model folder (fold counts differ by event).

## Catalog format

CSV files under `data/catalogs/` must provide at least:

| Column | Description |
|--------|-------------|
| `time` | Event origin time (parsable by pandas, e.g. `2000/1/2 2:48`) |
| `mag` | Magnitude |
| `latitude` | Latitude in degrees |
| `longitude` | Longitude in degrees |
| `depth` | Depth in km |

Extra columns are ignored. A `magnitude` column is accepted as an alias of `mag`.

## Test feature format

Inference inputs (e.g. `data/samples/test_jiuzhaigou.csv`) must contain:

```text
std_depthEQ, std_intertime, std_lat, std_lon, std_magnitude, std_energy_release
```

Each row is one day (or analysis step). Sliding windows of length 30 are built internally.

## Spatially varying prior dictionary (`p0_dict`)

Training uses a discrete prior key (`p0_label`) mapped to probabilities in `seed/config.py` (`P0_DICT`). These values encode region / event-class base rates used inside `SVP_BalancedFocalLoss`. If a training CSV lacks `p0_label`, the trainer defaults labels to key `5`.

---

## `scripts/prepare_train_features.py`

Builds positive observables around M≥`m1` events and random negative samples.

**Key options**

| Option | Default | Meaning |
|--------|---------|---------|
| `--catalog` | `data/catalogs/2000-2013.csv` | Input catalog |
| `--output-dir` | `data/processed` | Where CSVs are written |
| `--m1` | `6.0` | Large-event magnitude |
| `--radial-distance` | `120` | Search radius (km) |
| `--window-preeq` | `365` | Pre-event window length (days) |
| `--random-nodes` | `10` | Number of negative CSVs |

**Outputs**

- `Observables.csv`
- `Observables_random_{k}.csv`

---

## `scripts/prepare_test_features.py`

Builds a feature series at a fixed epicenter around a mainshock.

**Required**: `--lat`, `--lon`  
**Useful**: `--mainshock-time YYYY-MM-DD` or `--mainshock-mag`, `--output`

**Output**: CSV with the six standardized feature columns used by `predict.py`.

---

## `scripts/train.py`

Leave-one-segment-out training of LSTM + Random Forest.

**Key options**

| Option | Default | Meaning |
|--------|---------|---------|
| `--train-dir` | `data/processed` | Directory with Observables*.csv |
| `--model-number` | `5` | Selects `Observables_random_{n}.csv` and output name prefix |
| `--output-dir` | `models` | Weight directory |
| `--epochs` | `500` | LSTM epochs per fold |
| `--lr` | `0.003` | Adam learning rate |

**Outputs** (per validation segment `i`)

- `model_{n}_event_{i}.pth`
- `RF_model_{n}_event_{i}.joblib`

**Expected behaviour**: prints train/val loss every 50 epochs and RF accuracy / AUC per fold. Requires CUDA or CPU; GPU strongly preferred.

---

## `scripts/predict.py`

Loads the ensemble of pretrained models for a case-study event and scores its feature CSV.

**Key options**

| Option | Default | Meaning |
|--------|---------|---------|
| `--event` | `jiuzhaigou` | Selects sample CSV, model folder, and `outputs/<event>/` |
| `--input` | from event | Override feature CSV |
| `--model-dir` | from event | Override weight directory |
| `--output-dir` | `outputs/<event>` | Results directory |
| `--model-index` | all | Optional filter on `model_{i}_*` |
| `--event-start` / `--event-end` | all | Optional filter on fold index `j` |
| `--no-contribution-plot` | off | Skip contribution figure |

**Outputs** (under `outputs/<event>/`)

| File | Content |
|------|---------|
| `window_statistics_cs.csv` | Min / max / mean probability (%) per window |
| `hn_features_with_labels.csv` | Averaged LSTM hidden states + mean probability |
| `all_features_contribution_and_derivative.png` | Per-feature contribution bars and derivatives |

**Expected behaviour**: all matching LSTM/RF pairs in the event model folder are loaded; at least one pair must exist.

---

## `scripts/plot_probability.py`

Reads `window_statistics_cs.csv` and writes `probability_curve.png` for an event.

| Option | Default |
|--------|---------|
| `--event` | `jiuzhaigou` |
| `--input` | `outputs/<event>/window_statistics_cs.csv` |
| `--output` | `outputs/<event>/probability_curve.png` |
| `--post-event-days` | from event config (usually `200`) |

The horizontal axis is days relative to the mainshock (0 = event day), assuming the series ends `post-event-days` after the event (matching the shipped Jiuzhaigou sample construction).

---

## `scripts/plot_hn_cosine.py`

Compute pairwise cosine similarity among LSTM hidden-state samples and plot an EQ vs NEQ heatmap. Samples are class-balanced before the similarity matrix is built.

```bash
# Merge Wenchuan + Jiuzhaigou HN features (after predict)
python scripts/plot_hn_cosine.py --events wenchuan jiuzhaigou

# Single event or custom CSV
python scripts/plot_hn_cosine.py --events jiuzhaigou
python scripts/plot_hn_cosine.py --input path/to/hn_merged.csv --label-column y_true
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--events` | none | Event names; reads `outputs/<event>/hn_features_with_labels.csv` |
| `--input` | none | One or more HN CSVs (overrides `--events`) |
| `--output` | `outputs/hn_cosine_<tag>.png` | Heatmap path |
| `--n-samples` | `100` | Samples drawn per class (NEQ / EQ) |
| `--label-column` | auto | `y_true` / `Pred_Label` / `label` |
| `--no-absolute` | off | Keep signed cosine values |

**Expected behaviour**: label classes must each contain at least `--n-samples` rows; output is a coolwarm heatmap with NEQ/EQ blocks separated by dashed lines.

## `scripts/plot_prior_map.py`

Computes a Poisson-based SVP grid and saves a map plus CSV.

| Option | Default |
|--------|---------|
| `--catalog` | `data/catalogs/2000-2013.csv` |
| `--output` | `outputs/svp_spatial_map.png` |
| `--csv-output` | `outputs/svp_grid.csv` |

---

## `scripts/plot_space_heatmaps.py`

Interpolates daily probability fields from files named:

```text
statistics_lat_{lat}_lon_{lon}.csv
```

with columns `Window_Index` and `Avg_Probability` (also accepts `Mean_EQ_Probability(%)`).

| Option | Default | Notes |
|--------|---------|-------|
| `--data-dir` | `results/samples` | Sample grids shipped for demo |
| `--fault-shapefile` | `data/faults/active_faults_study_region.shp` | Fault overlay; needs cartopy + geopandas |
| `--no-faults` | off | Disable the fault overlay |
| `--max-days` | `5` | Use `0` to plot all days |

Full publication grids are not shipped; place your complete set of CSVs in a directory and point `--data-dir` to it.

---

## `scripts/plot_depth_sections.py`

Reads `statistics_depth_{depth}_lat_{lat}_lon_{lon}.csv` and writes depth–longitude / depth–latitude sections. Requires a user-supplied `--data-dir` (depth grids are not included in the sample pack).

---

## Troubleshooting

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `FileNotFoundError` for models | Wrong working directory | Run from repo root |
| Missing feature columns | Wrong CSV schema | Check column names in this guide |
| Cartopy import error | Optional deps missing | Omit `--fault-shapefile` or install cartopy/geopandas |
| `monotonic_cst` / RF unpickle error | scikit-learn ≥ 1.4 | `pip install "scikit-learn>=1.3,<1.4"` |
| Empty training features | Catalog / magnitude filters | Relax `--m1` or check catalog coverage |
