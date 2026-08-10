"""Feature engineering from earthquake catalogs."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def load_catalog(path: str | Path) -> pd.DataFrame:
    """
    Load a catalog CSV with columns: time, mag, latitude, longitude, depth.
    Extra columns are ignored. Time is parsed to datetime.
    """
    path = Path(path)
    df = pd.read_csv(path)
    # Support both named columns and positional first-five layout
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    for target in ("time", "mag", "latitude", "longitude", "depth"):
        if target in cols:
            rename[cols[target]] = target
        elif target == "mag" and "magnitude" in cols:
            rename[cols["magnitude"]] = "mag"
    if rename:
        df = df.rename(columns=rename)

    required = ["time", "mag", "latitude", "longitude", "depth"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        # Fall back to first five columns
        df = df.iloc[:, :5].copy()
        df.columns = required

    df = df[required].copy()
    df["time"] = pd.to_datetime(df["time"])
    df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
    df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
    df = df.sort_values("time").reset_index(drop=True)
    start_date = df["time"].min()
    df["days_since"] = (df["time"] - start_date).dt.total_seconds() / 86400.0
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (Earth radius 6370 km)."""
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)
    return (
        2
        * 6370
        * np.arcsin(
            np.sqrt(
                np.sin((lat2 * np.pi / 180 - lat1 * np.pi / 180) / 2) ** 2
                + np.cos(lat1 * np.pi / 180)
                * np.cos(lat2 * np.pi / 180)
                * np.sin((lon2 * np.pi / 180 - lon1 * np.pi / 180) / 2) ** 2
            )
        )
    )


def _zscore_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns and df[col].std() not in (0, None) and not np.isnan(df[col].std()):
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df


def compute_large_eq_features(
    combined_data: pd.DataFrame,
    large_eqs: pd.DataFrame,
    M1: float = 6,
    Nb: int = 365,
    Nf: int = 10,
    radial_distance: float = 120,
    window_preEQ: int = 365,
    min_mag: float = 1,
    max_mag: float = 6,
    step: int = 1,
) -> pd.DataFrame:
    """Build pre-mainshock observable time series around large events."""
    start_date = combined_data["time"].min()
    features = []

    for _, large_eq in large_eqs.iterrows():
        latitude_m1 = large_eq["latitude"]
        longitude_m1 = large_eq["longitude"]
        eq_date = large_eq["time"]

        distances = haversine_km(
            latitude_m1, longitude_m1, combined_data["latitude"], combined_data["longitude"]
        )
        eq_in_radius = combined_data[distances < radial_distance]
        eq_in_window = eq_in_radius[
            (eq_in_radius["mag"] >= min_mag) & (eq_in_radius["mag"] < max_mag)
        ]
        if eq_in_window.empty or (eq_date - eq_in_window["time"].min()).days < window_preEQ:
            continue

        day2 = pd.date_range(
            end=eq_date - timedelta(days=step),
            periods=int(window_preEQ / step),
            freq="D",
        )
        eq2 = pd.DataFrame({"day": day2, "days_to_EQ": (eq_date - day2).days})
        eq2["binary_variable"] = 0
        eq2.loc[eq2.index[-Nf:], "binary_variable"] = 1

        def _window(x):
            return eq_in_window[
                (eq_in_window["time"] >= x - timedelta(days=Nb)) & (eq_in_window["time"] < x)
            ]

        eq2["num_EQ"] = eq2["day"].apply(lambda x: _window(x).shape[0])
        eq2["std_depthEQ"] = eq2["day"].apply(lambda x: _window(x)["depth"].std())
        eq2["std_intertime"] = eq2["day"].apply(lambda x: _window(x)["days_since"].diff().std())
        eq2["std_lat"] = eq2["day"].apply(lambda x: _window(x)["latitude"].std())
        eq2["std_lon"] = eq2["day"].apply(lambda x: _window(x)["longitude"].std())
        eq2["std_magnitude"] = eq2["day"].apply(lambda x: _window(x)["mag"].std())
        eq2["std_energy_release"] = eq2["day"].apply(
            lambda x: _window(x)["mag"].apply(lambda m: 10 ** (1.5 * m + 11.8)).std()
        )
        eq2["date_large_EQ"] = eq_date.strftime("%Y-%m-%d %H:%M:%S")
        eq2 = eq2.dropna()
        eq2 = _zscore_cols(
            eq2,
            [
                "std_depthEQ",
                "std_intertime",
                "std_lat",
                "std_lon",
                "std_magnitude",
                "std_energy_release",
            ],
        )
        features.append(eq2)

    if not features:
        return pd.DataFrame()
    out = pd.concat(features, ignore_index=True)
    out["days_since"] = (out["day"] - start_date).dt.days
    return out


def compute_random_eq_features(
    combined_data: pd.DataFrame,
    large_eqs: pd.DataFrame,
    Nb: int = 365,
    window_preEQ: int = 365,
    radial_distance: float = 120,
    min_mag: float = 1,
    max_mag: float = 6,
    steps: int = 1,
    num_nodes: int = 1,
    lat_range=(28.0, 35.0),
    lon_range=(101.0, 108.0),
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Sample negative (no large event) locations/times for training."""
    rng = rng or np.random.default_rng(1234)
    start_date = combined_data["time"].min()
    features = []
    min_time = combined_data["days_since"].min()
    max_time = combined_data["days_since"].max()

    for _ in range(num_nodes):
        time = float(rng.uniform(min_time + 3 * 365, max_time + 3 * 365))
        latitude = float(rng.uniform(lat_range[0] + 1, lat_range[1] - 1))
        longitude = float(rng.uniform(lon_range[0] + 1, lon_range[1] - 1))

        if any(
            (abs(time - large_eqs["days_since"]) < 365)
            & (abs(latitude - large_eqs["latitude"]) < 0.5)
            & (abs(longitude - large_eqs["longitude"]) < 0.5)
        ):
            continue

        distances = haversine_km(
            latitude, longitude, combined_data["latitude"], combined_data["longitude"]
        )
        eq_in_radius = combined_data[
            (distances < radial_distance)
            & (combined_data["mag"] >= min_mag)
            & (combined_data["mag"] <= max_mag)
            & (combined_data["days_since"] >= time - window_preEQ)
            & (combined_data["days_since"] < time)
        ]
        if eq_in_radius.empty:
            continue

        day2 = pd.date_range(
            end=start_date + timedelta(days=int(time) - steps),
            periods=int(window_preEQ / steps),
            freq="D",
        )
        earthquakes2 = pd.DataFrame(
            {"day": day2, "days_to_EQ": (start_date + timedelta(days=int(time)) - day2).days}
        )
        earthquakes2["binary_variable"] = 0

        def _window(x):
            return eq_in_radius[
                (eq_in_radius["time"] >= (x - pd.Timedelta(days=Nb))) & (eq_in_radius["time"] < x)
            ]

        earthquakes2["num_EQ"] = earthquakes2["day"].apply(lambda x: _window(x).shape[0])
        earthquakes2["std_depthEQ"] = earthquakes2["day"].apply(lambda x: _window(x)["depth"].std())
        earthquakes2["std_intertime"] = earthquakes2["day"].apply(
            lambda x: _window(x)["days_since"].diff().std()
        )
        earthquakes2["std_lat"] = earthquakes2["day"].apply(lambda x: _window(x)["latitude"].std())
        earthquakes2["std_lon"] = earthquakes2["day"].apply(lambda x: _window(x)["longitude"].std())
        earthquakes2["std_magnitude"] = earthquakes2["day"].apply(lambda x: _window(x)["mag"].std())
        earthquakes2["std_energy_release"] = earthquakes2["day"].apply(
            lambda x: _window(x)["mag"].apply(lambda m: 10 ** (1.5 * m + 11.8)).std()
        )
        earthquakes2["date_large_EQ"] = (start_date + timedelta(days=int(time))).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        earthquakes2 = earthquakes2.dropna()
        earthquakes2 = _zscore_cols(
            earthquakes2,
            ["std_depthEQ", "std_intertime", "std_lat", "std_lon", "std_magnitude"],
        )
        features.append(earthquakes2)

    if not features:
        return pd.DataFrame()
    return pd.concat(features, ignore_index=True)


def compute_test_features_at_point(
    earthquakes: pd.DataFrame,
    latitude: float,
    longitude: float,
    time_m1: float,
    M1: float = 7.0,
    Nb: int = 365,
    Ns: int = 0,
    radial_distance: float = 120,
    window_preEQ: int = 365,
    steps: int = 1,
    min_mag: float = 1,
    max_mag: float = 6,
    day_start: int = -394,
    day_end: int = 200,
) -> pd.DataFrame:
    """
    Build a day-by-day feature series around a target location and mainshock time.

    Parameters
    ----------
    earthquakes : catalog with columns date/days/latitude/longitude/depth/magnitude
    time_m1 : days since catalog start of the target mainshock
    """
    rows = []
    for days_to_from_eq in range(day_start, day_end + 1, 1):
        time = time_m1 + days_to_from_eq
        distances = haversine_km(
            latitude, longitude, earthquakes["latitude"], earthquakes["longitude"]
        )
        earthquakes_aux = earthquakes[
            (distances < radial_distance)
            & (earthquakes["magnitude"] >= min_mag)
            & (earthquakes["magnitude"] < max_mag)
        ]
        before = earthquakes_aux[
            (earthquakes_aux["days"] < time)
            & (earthquakes_aux["days"] >= time - window_preEQ - Nb)
        ]["magnitude"]
        if time < window_preEQ or any(before >= M1):
            continue

        days2 = np.arange(time - window_preEQ, time, steps)
        earthquakes2 = pd.DataFrame(
            {"days2": days2, "days3": days2 - time, "binary_variable": 1}
        )
        for j, day in enumerate(days2):
            aux = earthquakes_aux[
                (earthquakes_aux["days"] >= day - Nb) & (earthquakes_aux["days"] < day)
            ].index
            if len(aux) >= Ns:
                earthquakes2.at[j, "num_EQ"] = len(aux)
                earthquakes2.at[j, "std_depthEQ"] = earthquakes_aux.loc[aux, "depth"].std()
                earthquakes2.at[j, "std_intertime"] = (
                    earthquakes_aux.loc[aux, "days"].diff().std()
                )
                earthquakes2.at[j, "std_lat"] = earthquakes_aux.loc[aux, "latitude"].std()
                earthquakes2.at[j, "std_lon"] = earthquakes_aux.loc[aux, "longitude"].std()
                earthquakes2.at[j, "std_magnitude"] = earthquakes_aux.loc[
                    aux, "magnitude"
                ].std()
                magnitudes = earthquakes_aux.loc[aux, "magnitude"]
                earthquakes2.at[j, "std_energy_release"] = magnitudes.apply(
                    lambda m: 10 ** (1.5 * m + 11.8)
                ).std()
            else:
                earthquakes2.iloc[j, 3:] = np.nan

        data_eq = earthquakes2.dropna()
        if data_eq.empty:
            continue
        data_eq = _zscore_cols(
            data_eq,
            [
                "std_depthEQ",
                "std_intertime",
                "std_lat",
                "std_lon",
                "std_magnitude",
                "std_energy_release",
            ],
        )
        last = data_eq.iloc[-1]
        rows.append(
            {
                "std_depthEQ": last["std_depthEQ"],
                "std_intertime": last["std_intertime"],
                "std_lat": last["std_lat"],
                "std_lon": last["std_lon"],
                "std_magnitude": last["std_magnitude"],
                "std_energy_release": last["std_energy_release"],
                "days_to_EQ": time - time_m1,
            }
        )

    return pd.DataFrame(rows)
