"""Step 3: make the forecast the website shows.

Downloads the latest weather forecast and river data, runs the saved model
for every beach for today and the next few days, and writes
docs/data/forecast.json.

Run:  python forecast.py
      python forecast.py --demo   (no internet: fills in made-up weather so you can test the site)
"""
import argparse
import json
from datetime import date, datetime

import joblib
import numpy as np
import pandas as pd

import features
from config import FORECAST_DAYS, MODELS, PROCESSED, REPORTS, WEB_DATA


def get_weather(sites, demo):
    if demo:
        rng = np.random.default_rng(int(date.today().strftime("%Y%m%d")))
        days = pd.date_range(pd.Timestamp.today().normalize() - pd.Timedelta(days=10),
                             periods=10 + FORECAST_DAYS)
        return pd.concat([pd.DataFrame({
            "date": days, "site_id": s["site_id"],
            "rain_mm": rng.exponential(6, len(days)) * (rng.random(len(days)) < 0.35),
            "temp_max_c": 29 + rng.normal(0, 2, len(days)),
        }) for s in sites], ignore_index=True)
    import weather
    return weather.forecast(sites)


def get_gauges(demo):
    if demo:
        return None
    import usgs
    start = (date.today() - pd.Timedelta(days=45)).isoformat()
    g = usgs.all_gauges(start)
    if g.empty:
        return None
    # River data only exists up to today, so carry the latest reading forward
    future = pd.date_range(g["date"].max(), pd.Timestamp.today().normalize() + pd.Timedelta(days=FORECAST_DAYS))
    return g.set_index("date").reindex(g["date"].tolist() + list(future[1:])).ffill().reset_index()


def label(p, q50, q90, cutoffs, threshold):
    if q50 >= threshold or p >= cutoffs["avoid"]:
        return "avoid"
    if q90 >= threshold or p >= cutoffs["caution"]:
        return "caution"
    return "low"


def inches(mm):
    return None if mm is None or pd.isna(mm) else round(float(mm) / 25.4, 2)


def day_record(r, bundle):
    return {
        "date": r["date"].date().isoformat(),
        "risk": round(float(r["p"]), 3),
        "level": label(r["p"], r["q50"], r["q90"], bundle["cutoffs"], bundle["threshold"]),
        "range": [round(float(r["q10"]), 1), round(float(r["q50"]), 1), round(float(r["q90"]), 1)],
        "rain_today_in": inches(r.get("rain_today")),
        "rain_prev_3d_in": inches(r.get("rain_prev_3d")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    bundle = joblib.load(MODELS / "model.joblib")
    sites = json.loads((PROCESSED / "sites.json").read_text())
    samples = pd.read_csv(PROCESSED / "samples.csv", parse_dates=["date"])

    wx = get_weather(sites, args.demo)
    gauges = get_gauges(args.demo)

    today = pd.Timestamp.today().normalize()
    days = pd.date_range(today, periods=FORECAST_DAYS)
    rows = pd.DataFrame([(s["site_id"], d) for s in sites for d in days], columns=["site_id", "date"])
    X = features.build(rows, wx, gauges)
    X = features.add_site_dummies(X, bundle["sites"])
    for c in bundle["features"]:
        if c not in X:
            X[c] = np.nan  # the model handles missing inputs
    X["p"] = bundle["risk"].predict_proba(X[bundle["features"]])[:, 1]
    for q, m in bundle["quantiles"].items():
        X[f"q{int(q*100)}"] = 10 ** m.predict(X[bundle["features"]])

    metrics_path = REPORTS / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    out_sites = []
    for s in sites:
        g = X[X.site_id == s["site_id"]].sort_values("date")
        last = samples[samples.site_id == s["site_id"]].sort_values("date").iloc[-1]
        out_sites.append({
            "id": s["site_id"], "name": s["name"], "lat": s["lat"], "lon": s["lon"],
            "samples": s["samples"], "exceedances": s["exceedances"],
            "last_sample": {"date": last["date"].date().isoformat(), "ecoli": float(last["ecoli"])},
            "days": [day_record(r, bundle) for r in g.to_dict("records")],
        })

    payload = {
        "generated": datetime.now().isoformat(timespec="minutes"),
        "demo": args.demo,
        "threshold": bundle["threshold"],
        "model_mode": bundle["mode"],
        "validation": {k: metrics.get(k) for k in ("primary", "primary_scores", "p_base", "range_model", "tested_samples")},
        "sites": out_sites,
    }
    (WEB_DATA / "forecast.json").write_text(json.dumps(payload, indent=2))
    print(f"Wrote forecast for {len(out_sites)} sites to {WEB_DATA / 'forecast.json'}")


if __name__ == "__main__":
    main()
