# Active fault polylines (study-region subset)

- **File**: `active_faults_study_region.shp` (+ `.shx`, `.dbf`, `.prj`, `.cpg`)
- **Content**: Active-fault line geometries clipped to approximately
  longitude 101–106°E and latitude 30.5–35.5°N (covers the spatial plotting domain).
  Attribute tables were dropped (geometry-only) to avoid encoding issues when sharing.
- **CRS**: EPSG:4326
- **Usage**: default overlay for `scripts/plot_space_heatmaps.py`

```bash
python scripts/plot_space_heatmaps.py
# disable overlay:
python scripts/plot_space_heatmaps.py --no-faults
```

If you redistribute this repository, confirm that sharing this clipped subset complies with the license/terms of your original national active-fault dataset.
