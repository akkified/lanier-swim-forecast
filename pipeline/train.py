"""Step 2: train and test the models.

Two models:
  1. Risk model: chance that E. coli is at or above the swim limit (235).
     Logistic regression and gradient boosting are both tested; the one with
     the better held-out score is saved.
  2. Range model: likely E. coli range (10th, 50th, 90th percentile).

Testing uses leave-one-season-out: train on every other summer, predict the
held-out summer, repeat. That is the honest way to check a forecast, because
the model never sees any data from the season it is graded on.

Run:  python train.py            (downloads weather and USGS data)
      python train.py --offline  (uses only the rainfall column in the spreadsheet)
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features
from config import MODELS, PROCESSED, REPORTS, THRESHOLD, WEB_DATA


def load_inputs(offline: bool):
    samples = pd.read_csv(PROCESSED / "samples.csv", parse_dates=["date"])
    sites = json.loads((PROCESSED / "sites.json").read_text())
    weather = gauges = None
    import local_inputs
    if not offline and local_inputs.available():
        weather, gauges = local_inputs.load()
        print(f"Using {local_inputs.INPUTS.name}: {len(weather)} weather rows, "
              f"USGS {[c for c in gauges.columns if c != 'date']}")
        return samples, sites, weather, gauges
    if not offline:
        import usgs
        import weather as wx
        start = (samples.date.min() - pd.Timedelta(days=45)).date().isoformat()
        end = samples.date.max().date().isoformat()
        try:
            weather = wx.history(sites, start, end)
            print(f"Weather rows: {len(weather)}")
        except Exception as e:  # noqa: BLE001
            print(f"Weather download failed ({e}); falling back to offline mode")
        try:
            gauges = usgs.all_gauges(start, end)
            print(f"USGS columns: {[c for c in gauges.columns if c != 'date']}")
        except Exception as e:  # noqa: BLE001
            print(f"USGS download failed ({e}); training without river data")
    return samples, sites, weather, gauges


def make_models():
    risk_gb = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=10,
        l2_regularization=1.0, class_weight="balanced", random_state=0)
    risk_lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000))
    # Linear quantile models: simple enough to run inside the web page
    quantiles = {q: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  QuantileRegressor(quantile=q, alpha=0.01, solver="highs"))
                 for q in (0.1, 0.5, 0.9)}
    return risk_gb, risk_lr, quantiles


def season_cv(df, cols):
    """Leave-one-season-out predictions for every sample."""
    df = df.copy()
    df["season"] = df["date"].dt.year
    preds = pd.DataFrame(index=df.index, columns=["p_gb", "p_lr", "p_base", "q10", "q50", "q90"], dtype=float)
    for season in sorted(df.season.unique()):
        test = df.season == season
        train = ~test
        if df.loc[train, "exceeds"].sum() < 3:
            continue
        gb, lr, qs = make_models()
        X_tr, y_tr = df.loc[train, cols], df.loc[train, "exceeds"]
        gb.fit(X_tr, y_tr)
        lr.fit(X_tr, y_tr)
        preds.loc[test, "p_gb"] = gb.predict_proba(df.loc[test, cols])[:, 1]
        preds.loc[test, "p_lr"] = lr.predict_proba(df.loc[test, cols])[:, 1]
        rate = df.loc[train].groupby("site_id")["exceeds"].mean()
        preds.loc[test, "p_base"] = df.loc[test, "site_id"].map(rate).fillna(y_tr.mean()).values
        for q, m in qs.items():
            m.fit(X_tr, df.loc[train, "log_ecoli"])
            preds.loc[test, f"q{int(q*100)}"] = m.predict(df.loc[test, cols])
    return df.join(preds)


def score(df):
    out = {}
    ok = df.dropna(subset=["p_gb"])
    y = ok["exceeds"]
    for name in ("p_base", "p_lr", "p_gb"):
        p = ok[name]
        top = p >= np.quantile(p, 0.9)  # flag the riskiest 10% of days
        out[name] = {
            "auc": round(roc_auc_score(y, p), 3) if y.nunique() > 1 else None,
            "brier": round(brier_score_loss(y, p), 4),
            "caught_in_top10pct": int((top & (y == 1)).sum()),
            "total_high": int(y.sum()),
        }
    inside = ((ok["log_ecoli"] >= ok["q10"]) & (ok["log_ecoli"] <= ok["q90"])).mean()
    out["range_model"] = {
        "share_inside_10_90_range": round(float(inside), 3),
        "median_abs_error_log10": round(float((ok["log_ecoli"] - ok["q50"]).abs().median()), 3),
    }
    out["tested_samples"] = int(len(ok))
    out["by_season"] = {int(s): {"n": int(len(g)), "high": int(g.exceeds.sum())}
                        for s, g in ok.groupby(ok.date.dt.year)}
    return out


def _linear_parts(pipe):
    imp, sc, model = pipe[0], pipe[1], pipe[-1]
    coef = model.coef_[0] if model.coef_.ndim == 2 else model.coef_
    intercept = float(np.ravel(model.intercept_)[0])
    return {"fill": imp.statistics_.tolist(), "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
            "coef": coef.tolist(), "intercept": intercept}


def export_for_web(lr, qs, cols, site_ids, cutoffs, metrics):
    """Save the models as plain numbers so docs/app.js can run them in the browser."""
    if lr is None:
        print("Main model is gradient boosting; web export skipped (browser version needs the linear model)")
        return
    samples = pd.read_csv(PROCESSED / "samples.csv", parse_dates=["date"])
    sites = json.loads((PROCESSED / "sites.json").read_text())
    for s in sites:
        last = samples[samples.site_id == s["site_id"]].sort_values("date").iloc[-1]
        s["last_sample"] = {"date": last["date"].date().isoformat(), "ecoli": float(last["ecoli"])}
    out = {
        "features": cols, "threshold": THRESHOLD, "cutoffs": cutoffs,
        "risk": _linear_parts(lr),
        "quantiles": {str(q): _linear_parts(m) for q, m in qs.items()},
        "sites": sites,
        "validation": {k: metrics.get(k) for k in ("primary", "primary_scores", "p_base", "range_model", "tested_samples")},
        "trained": pd.Timestamp.today().date().isoformat(),
    }
    (WEB_DATA / "model.json").write_text(json.dumps(out, indent=1))
    print(f"Saved browser model to {WEB_DATA / 'model.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    samples, sites, weather, gauges = load_inputs(args.offline)
    offline = weather is None
    if offline:
        # Fallback: only the rain the field team wrote down on sample day
        samples["rain_today"] = samples["rain_field"] * 25.4
    site_ids = [s["site_id"] for s in sites]

    df = features.build(samples, weather, gauges)
    df = features.add_site_dummies(df, site_ids)
    cols = features.feature_columns(df, site_ids)
    print(f"{len(cols)} features: {cols}")

    cv = season_cv(df, cols)
    metrics = score(cv)
    metrics["mode"] = "offline" if offline else "online"
    metrics["features"] = cols
    print(json.dumps({k: v for k, v in metrics.items() if k != "features"}, indent=2))

    # Pick the risk model that did best on held-out seasons
    primary = max(("p_lr", "p_gb"), key=lambda k: metrics[k]["auc"] or 0)
    metrics["primary"] = primary
    metrics["primary_scores"] = metrics[primary]
    print(f"Main risk model: {'logistic regression' if primary == 'p_lr' else 'gradient boosting'}")

    gb, lr, qs = make_models()
    risk = lr if primary == "p_lr" else gb
    risk.fit(df[cols], df["exceeds"])
    for m in qs.values():
        m.fit(df[cols], df["log_ecoli"])
    # Cutoffs for the app's labels, set from the held-out predictions:
    # top 10% of risk scores = avoid, next 15% = caution
    p = cv[primary].dropna()
    cutoffs = {"avoid": float(p.quantile(0.90)), "caution": float(p.quantile(0.75))}
    metrics["cutoffs"] = cutoffs

    joblib.dump({"risk": risk, "quantiles": qs, "features": cols, "sites": site_ids,
                 "threshold": THRESHOLD, "mode": metrics["mode"], "cutoffs": cutoffs},
                MODELS / "model.joblib")

    export_for_web(risk if primary == "p_lr" else None, qs, cols, site_ids, cutoffs, metrics)

    cv.to_csv(REPORTS / "cv_predictions.csv", index=False)
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model to {MODELS / 'model.joblib'}")


if __name__ == "__main__":
    main()
