"""Turn weather, river, and lake data into model inputs for each (site, day).

Only inputs that are known ahead of time are used, so the same features
work for past samples (training) and future days (forecast).
"""
import numpy as np
import pandas as pd

from config import RAIN_WINDOWS

BIG_RAIN_MM = 12.7  # half an inch


def weather_features(weather: pd.DataFrame) -> pd.DataFrame:
    """weather: site_id, date, rain_mm, temp_max_c (one row per site per day)."""
    out = []
    for site, g in weather.sort_values("date").groupby("site_id"):
        g = g.set_index("date").asfreq("D")
        g["site_id"] = site
        rain = g["rain_mm"]
        f = pd.DataFrame(index=g.index)
        f["site_id"] = site
        f["rain_today"] = rain
        prev = rain.shift(1)  # rain on earlier days only
        for w in RAIN_WINDOWS:
            f[f"rain_prev_{w}d"] = prev.rolling(w, min_periods=1).sum()
        big = (rain >= BIG_RAIN_MM)
        last_big = pd.Series(np.where(big, np.arange(len(g)), np.nan), index=g.index).ffill()
        f["days_since_big_rain"] = (np.arange(len(g)) - last_big).clip(upper=30)
        f["temp_max_3d"] = g["temp_max_c"].rolling(3, min_periods=1).mean()
        out.append(f.reset_index())
    return pd.concat(out, ignore_index=True)


def gauge_features(gauges: pd.DataFrame) -> pd.DataFrame:
    """gauges: date plus any of chestatee_flow, chattahoochee_flow, lake_level."""
    if gauges.empty or len(gauges.columns) == 1:
        return pd.DataFrame(columns=["date"])
    g = gauges.sort_values("date").set_index("date").asfreq("D")
    f = pd.DataFrame(index=g.index)
    for col in ("chestatee_flow", "chattahoochee_flow"):
        if col in g:
            prev = g[col].shift(1)
            f[f"{col}_log"] = np.log10(prev.clip(lower=1))
            # flow compared to the past month: above 1 means the river is running high
            f[f"{col}_vs_30d"] = prev / prev.rolling(30, min_periods=7).median()
    if "lake_level" in g:
        prev = g["lake_level"].shift(1)
        f["lake_level"] = prev
        f["lake_level_change_7d"] = prev - prev.shift(7)
    return f.reset_index()


def calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    doy = df["date"].dt.dayofyear
    df["season_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["season_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    return df


def build(rows: pd.DataFrame, weather: pd.DataFrame | None, gauges: pd.DataFrame | None) -> pd.DataFrame:
    """rows: at least site_id and date. Returns rows with every feature attached."""
    rows = rows.copy()
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    if weather is not None and not weather.empty:
        rows = rows.merge(weather_features(weather), on=["site_id", "date"], how="left")
    if gauges is not None and not gauges.empty:
        rows = rows.merge(gauge_features(gauges), on="date", how="left")
    return calendar_features(rows)


def feature_columns(df: pd.DataFrame, sites: list[str]) -> list[str]:
    skip = {"site_id", "date", "rain_field"}  # rain_field is only known on sample days
    base = [c for c in df.columns if c.startswith(("rain_", "days_since", "temp_", "chestatee",
                                                   "chattahoochee", "lake_level", "season", "weekend"))
            and c not in skip]
    return base + [f"site_{s}" for s in sites]


def add_site_dummies(df: pd.DataFrame, sites: list[str]) -> pd.DataFrame:
    for s in sites:
        df[f"site_{s}"] = (df["site_id"] == s).astype(int)
    return df
