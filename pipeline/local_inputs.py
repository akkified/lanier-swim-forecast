"""Load weather and USGS data from data/raw/lanier_inputs.json.

That file was downloaded once in a browser (Open-Meteo archive + USGS daily
values, 2019 to Sept 2026), so training works without internet access.
"""
import json

import pandas as pd

from config import ROOT

INPUTS = ROOT / "data" / "raw" / "lanier_inputs.json"


def available() -> bool:
    return INPUTS.exists()


def load():
    raw = json.loads(INPUTS.read_text())
    weather = pd.concat([
        pd.DataFrame({"date": pd.to_datetime(w["time"]), "rain_mm": w["rain_mm"],
                      "temp_max_c": w["temp_max_c"], "site_id": site})
        for site, w in raw["weather"].items()
    ], ignore_index=True)

    gauges = None
    for name, g in raw["usgs"].items():
        df = pd.DataFrame({"date": pd.to_datetime(g["date"]), name: pd.to_numeric(g["value"], errors="coerce")})
        df.loc[df[name] < 0, name] = None  # USGS uses negative codes for missing values
        df = df.dropna().drop_duplicates("date").sort_values("date")
        gauges = df if gauges is None else gauges.merge(df, on="date", how="outer")
    return weather, gauges.sort_values("date").reset_index(drop=True)
