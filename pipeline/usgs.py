"""Daily river flow and lake level from USGS.

Tries the newer USGS Water Data API first, then the older WaterServices API.
Everything here is optional. If both fail, the model just skips these inputs.
"""
import json
from datetime import date

import pandas as pd
import requests

from config import CACHE, USGS_GAUGES

NEW_API = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items"
OLD_API = "https://waterservices.usgs.gov/nwis/dv/"


def _new_api(site: str, param: str, start: str, end: str) -> pd.DataFrame:
    rows, url = [], NEW_API
    params = {
        "monitoring_location_id": f"USGS-{site}", "parameter_code": param,
        "statistic_id": "00003", "datetime": f"{start}/{end}", "f": "json", "limit": 10000,
    }
    while url:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        rows += [(f["properties"]["time"], f["properties"]["value"]) for f in data.get("features", [])]
        nxt = [l["href"] for l in data.get("links", []) if l.get("rel") == "next"]
        url, params = (nxt[0], None) if nxt else (None, None)
    return pd.DataFrame(rows, columns=["date", "value"])


def _old_api(site: str, param: str, start: str, end: str) -> pd.DataFrame:
    r = requests.get(OLD_API, params={
        "format": "json", "sites": site, "parameterCd": param,
        "startDT": start, "endDT": end,
    }, timeout=60)
    r.raise_for_status()
    series = r.json()["value"]["timeSeries"]
    if not series:
        return pd.DataFrame(columns=["date", "value"])
    vals = series[0]["values"][0]["value"]
    return pd.DataFrame([(v["dateTime"], v["value"]) for v in vals], columns=["date", "value"])


def daily(name: str, start: str, end: str | None = None) -> pd.DataFrame:
    site, param = USGS_GAUGES[name]
    end = end or date.today().isoformat()
    cache = CACHE / f"usgs_{name}_{start}_{end}.json"
    if cache.exists():
        df = pd.DataFrame(json.loads(cache.read_text()))
    else:
        df = None
        for fetch in (_new_api, _old_api):
            try:
                df = fetch(site, param, start, end)
                if len(df):
                    break
            except Exception as e:  # noqa: BLE001
                print(f"  USGS {name} via {fetch.__name__} failed: {e}")
        if df is None or df.empty:
            return pd.DataFrame(columns=["date", name])
        cache.write_text(df.to_json(orient="records", date_format="iso"))
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df[name] = pd.to_numeric(df["value"], errors="coerce")
    return df[["date", name]].dropna().drop_duplicates("date")


def all_gauges(start: str, end: str | None = None) -> pd.DataFrame:
    out = None
    for name in USGS_GAUGES:
        df = daily(name, start, end)
        if df.empty:
            continue
        out = df if out is None else out.merge(df, on="date", how="outer")
    return out if out is not None else pd.DataFrame(columns=["date"])
