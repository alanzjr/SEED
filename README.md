# Anomaly detection of low-magnitude seismic activity prior to major earthquakes via the SEED framework

This repository contains the open-source implementation accompanying the manuscript. The code builds Spatially Varying Prior (SVP)–aware features from earthquake catalogs, trains an LSTM feature extractor with a Random Forest classifier, and produces temporal probability curves and optional spatial / depth visualizations.

## Computer Code Availability

- **Name of code**: SEED (SVP-enhanced earthquake forecasting demo)
- **Developer / contact**: Jiarui Zhang
- **Year first available**: 2026
- **Hardware required**: Standard workstation; GPU optional for training
- **Software required**: Python 3.9+ (tested conceptually with PyTorch 1.13+/2.x)
- **Program language**: Python
- **Program size**: Source < 1 MB; pretrained models ~35 MB; catalogs ~13 MB
- **License**: MIT (see [LICENSE](LICENSE))

## Repository layout

```text
SEED/
├── LICENSE
├── README.md
├── requirements.txt
├── main.py               # One-command entry for reproducing main results
├── docs/
│   ├── USER_GUIDE.md
│   └── TUTORIAL.md
├── seed/                 # Importable Python package
├── scripts/              # Individual command-line tools
├── data/
│   ├── catalogs/         # Raw earthquake catalogs
│   ├── faults/           # Study-region active-fault shapefile (for map overlays)
│   ├── processed/        # Generated training features (not shipped)
│   └── samples/          # Test features for each case-study event
├── models/
│   ├── jiuzhaigou/       # Pretrained ensembles per event
│   ├── wenchuan/
│   ├── lushan/
│   ├── lushan2022/
│   ├── luding/
│   └── jishishan/
├── results/samples/      # Small spatial-grid CSV examples
└── outputs/              # Runtime outputs (created on demand, per event)
```

## Installation

```bash
# from the repository root
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

**Notes**

- Core reproduction (predict + time-series plot) needs: `numpy`, `pandas`, `torch`, `scikit-learn`, `joblib`, `matplotlib`.
- Install **scikit-learn 1.3.x** (`scikit-learn>=1.3,<1.4` as pinned in `requirements.txt`). The shipped Random Forest `.joblib` files were serialized with 1.3.2; scikit-learn 1.4+ may fail to unpickle them.
- Spatial basemaps with fault overlays optionally need `cartopy` and `geopandas`.
- Training benefits from a CUDA-capable GPU but also runs on CPU (slower).

## Quick start (reproduce main temporal result)

No retraining is required. From the repository root:

```bash
python main.py                          # default: Jiuzhaigou
python main.py --event wenchuan         # one of the other case studies
python main.py --event all              # all six events
```

Supported `--event` values: `jiuzhaigou`, `wenchuan`, `lushan`, `lushan2022`, `luding`, `jishishan`.

Equivalent explicit commands:

```bash
python scripts/predict.py --event wenchuan
python scripts/plot_probability.py --event wenchuan
```

Optional modes:

```bash
python main.py --mode prior    # SVP prior map
python main.py --mode space    # sample spatial heatmaps
python main.py --mode all      # demo + prior + space
```

Expected outputs in `outputs/<event>/`:

- `window_statistics_cs.csv` — ensemble min / max / mean probabilities
- `hn_features_with_labels.csv` — mean LSTM hidden features and labels
- `all_features_contribution_and_derivative.png` — feature contribution panels
- `probability_curve.png` — probability vs. days relative to the mainshock

## Typical workflow

1. **(Optional) Prepare training features** from a catalog  
   `python scripts/prepare_train_features.py --catalog data/catalogs/2000-2013.csv`
2. **(Optional) Train**  
   `python scripts/train.py --train-dir data/processed --model-number 5`
3. **Use a shipped test feature CSV** under `data/samples/`  
   e.g. `test_jiuzhaigou.csv`, `test_wenchuan.csv`, …
4. **Predict** with the matching ensemble under `models/<event>/`  
   `python scripts/predict.py --event wenchuan`
5. **Plot**  
   `python scripts/plot_probability.py --event wenchuan`
6. **Optional SVP map**  
   `python scripts/plot_prior_map.py`
7. **Optional spatial demo** (uses the few CSVs under `results/samples/`; overlays faults from `data/faults/` by default)  
   `python scripts/plot_space_heatmaps.py --max-days 5`
8. **Optional HN cosine similarity** (e.g. Wenchuan + Jiuzhaigou)  
   ```bash
   python scripts/predict.py --event wenchuan --no-contribution-plot
   python scripts/predict.py --event jiuzhaigou --no-contribution-plot
   python scripts/plot_hn_cosine.py --events wenchuan jiuzhaigou
   ```

See [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for inputs/outputs/parameters and [docs/TUTORIAL.md](docs/TUTORIAL.md) for step-by-step examples.

## Data notes

- Catalogs are provided as CSV files with columns `time,mag,latitude,longitude,depth`.
- Full dense spatial grids used for publication figures are large; this repository ships a **small sample** under `results/samples/` so reviewers can exercise the plotting code. To regenerate a full grid, run prediction at each `(lat, lon)` of interest and write `statistics_lat_{lat}_lon_{lon}.csv` files with columns `Window_Index,Min_Probability,Max_Probability,Avg_Probability`.

## Catalog enhancement

The earthquake catalog enhancement used in this study was performed using the Earthquake Rescaled Aftershock Seismicity (ERAS) model developed by Rundle et al.
The ERAS implementation is publicly available from Zenodo: Rundle, J. (2024). Earthquake_Rescaled_Aftershock_Seismicity_ERAS
[Software]. Zenodo. https://doi.org/10.5281/zenodo.10810321
The methodological details of the ERAS model are described in: Rundle, J. B., Baughman, I., & Zhang, T. (2024). Nowcasting Earthquakes With Stochastic Simulations: Information Entropy of Earthquake Catalogs. Earth and Space Science, 11, e2023EA003367. https://doi.org/10.1029/2023EA003367


## License

MIT License — see [LICENSE](LICENSE).
