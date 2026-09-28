"""Rainfall and temperature from Open-Meteo (free, no API key).

- history(): past daily weather for each site, used for training
- forecast(): recent past plus the next few days, used for the live app
Results are cached in data/cache so you only download once.
"""
import json
from datetime import date

import pandas as pd
import requests

from config import CACHE, FORECAST_DAYS

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DAILY = "precipitation_sum,temperature_2m_max"


def _to_frame(payload: dict, site_id: str) -> pd.DataFrame:
    d = payload["daily"]
    df = pd.DataFrame({
        "date": pd.to_datetime(d["time"]),
        "rain_mm": d["precipitation_sum"],
        "temp_max_c": d["temperature_2m_max"],
    })
    df["site_id"] = site_id
    return df


def history(sites: list[dict], start: str, end: str | None = None) -> pd.DataFrame:
    end = end or date.today().isoformat()
    frames = []
    for s in sites:
        cache = CACHE / f"weather_{s['site_id']}_{start}_{end}.json"
        if cache.exists():
            payload = json.loads(cache.read_text())
        else:
            r = requests.get(ARCHIVE_URL, params={
                "latitude": s["lat"], "longitude": s["lon"],
                "start_date": start, "end_date": end,
                "daily": DAILY, "timezone": "America/New_York",
            }, timeout=60)
            r.raise_for_status()
            payload = r.json()
            cache.write_text(json.dumps(payload))
        frames.append(_to_frame(payload, s["site_id"]))
    return pd.concat(frames, ignore_index=True)


def forecast(sites: list[dict], past_days: int = 10) -> pd.DataFrame:
    frames = []
    for s in sites:
        r = requests.get(FORECAST_URL, params={
            "latitude": s["lat"], "longitude": s["lon"],
            "daily": DAILY, "timezone": "America/New_York",
            "past_days": past_days, "forecast_days": FORECAST_DAYS,
        }, timeout=60)
        r.raise_for_status()
        frames.append(_to_frame(r.json(), s["site_id"]))
    return pd.concat(frames, ignore_index=True)
