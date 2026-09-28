"""Check that docs/model.js gives the same answers as the Python model.

Picks several real past days, computes predictions in Python, then runs the
same inputs through model.js with Node and compares.
Run from the cac folder:  python tests/check_browser_model.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import features  # noqa: E402
import local_inputs  # noqa: E402

model = json.loads((ROOT / "docs" / "data" / "model.json").read_text())
weather, gauges = local_inputs.load()
site_ids = [s["site_id"] for s in model["sites"]]
check_days = ["2019-07-16", "2023-06-01", "2025-05-15", "2025-07-31", "2026-08-20"]


def py_predict(row):
    def lin(parts):
        x = np.array([row[c] if pd.notna(row[c]) else f for c, f in zip(model["features"], parts["fill"])])
        return parts["intercept"] + np.sum(np.array(parts["coef"]) * (x - parts["mean"]) / parts["scale"])
    return 1 / (1 + np.exp(-lin(model["risk"]))), {k: 10 ** lin(v) for k, v in model["quantiles"].items()}


cases, expected = [], []
for day in check_days:
    d = pd.Timestamp(day)
    lo = d - pd.Timedelta(days=60)
    w = weather[(weather.date >= lo) & (weather.date <= d)]
    g = gauges[(gauges.date >= lo) & (gauges.date <= d)]
    rows = pd.DataFrame({"site_id": site_ids, "date": d})
    X = features.add_site_dummies(features.build(rows, w, g), site_ids)
    for c in model["features"]:
        if c not in X:
            X[c] = np.nan
    for r in X.to_dict("records"):
        p, q = py_predict(r)
        expected.append({"site": r["site_id"], "date": day, "risk": p, "q50": q["0.5"]})
    cases.append({
        "date": day,
        "weather": {s: {"dates": [x.date().isoformat() for x in ws.date], "rain_mm": ws.rain_mm.tolist(),
                        "temp_max_c": ws.temp_max_c.tolist()} for s, ws in w.groupby("site_id")},
        "gauges": {c: [{"date": x.date().isoformat(), "value": v} for x, v in zip(g.date, g[c]) if pd.notna(v)]
                   for c in g.columns if c != "date"},
        "first": lo.date().isoformat(),
    })

js = """
const M = require('%s');
const model = require('%s');
const cases = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const out = [];
for (const c of cases) {
  const gf = M.gaugeFeatures(M.dailyGauges(c.gauges, c.first, c.date));
  for (const s of model.sites.map(x => x.site_id)) {
    const wf = M.weatherFeatures(c.weather[s]);
    const r = M.predict(model, M.buildRow(s, c.date, wf, gf, model.sites.map(x => x.site_id)));
    out.push({site: s, date: c.date, risk: r.risk, q50: r.range[1]});
  }
}
console.log(JSON.stringify(out));
""" % (ROOT / "docs" / "model.js", ROOT / "docs" / "data" / "model.json")
res = subprocess.run(["node", "-e", js], input=json.dumps(cases), capture_output=True, text=True, check=True)
got = json.loads(res.stdout)

worst = 0.0
for e, g in zip(expected, got):
    assert e["site"] == g["site"] and e["date"] == g["date"]
    diff = abs(e["risk"] - g["risk"])
    worst = max(worst, diff)
    if diff > 1e-6:
        print(f"MISMATCH {e['date']} {e['site']}: python {e['risk']:.4f} js {g['risk']:.4f}")
print(f"Compared {len(got)} site-days. Largest risk difference: {worst:.2e}")
print("PASS" if worst < 1e-6 else "FAIL")
