// Runs the trained model in the browser. This mirrors pipeline/features.py and
// the saved logistic regression / quantile models in docs/data/model.json.
// Works in the browser (window.LanierModel) and in Node (module.exports) so it can be tested.

(function (root) {
  const BIG_RAIN_MM = 12.7;
  const RAIN_WINDOWS = [1, 2, 3, 7];

  const isNum = (v) => v !== null && v !== undefined && !Number.isNaN(v);

  function addDays(iso, n) {
    const d = new Date(iso + "T00:00:00Z");
    d.setUTCDate(d.getUTCDate() + n);
    return d.toISOString().slice(0, 10);
  }

  function dayOfYear(iso) {
    const d = new Date(iso + "T00:00:00Z");
    return Math.round((d - Date.UTC(d.getUTCFullYear(), 0, 1)) / 86400000) + 1;
  }

  // Sum / mean / median of the numbers in a window, skipping missing values (like pandas rolling)
  function windowStat(arr, end, size, minCount, fn) {
    const vals = [];
    for (let i = Math.max(0, end - size + 1); i <= end; i++) if (isNum(arr[i])) vals.push(arr[i]);
    return vals.length >= minCount ? fn(vals) : null;
  }
  const sum = (v) => v.reduce((a, b) => a + b, 0);
  const mean = (v) => sum(v) / v.length;
  const median = (v) => {
    const s = [...v].sort((a, b) => a - b);
    const m = Math.floor(s.length / 2);
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  };

  // weather: {dates: [...], rain_mm: [...], temp_max_c: [...]}, consecutive days
  function weatherFeatures(weather) {
    const n = weather.dates.length;
    const rain = weather.rain_mm;
    const prev = [null, ...rain.slice(0, n - 1)];
    const out = {};
    let lastBig = null;
    weather.dates.forEach((date, i) => {
      if (isNum(rain[i]) && rain[i] >= BIG_RAIN_MM) lastBig = i;
      const f = { rain_today: isNum(rain[i]) ? rain[i] : null };
      for (const w of RAIN_WINDOWS) f[`rain_prev_${w}d`] = windowStat(prev, i, w, 1, sum);
      f.days_since_big_rain = lastBig === null ? null : Math.min(i - lastBig, 30);
      f.temp_max_3d = windowStat(weather.temp_max_c, i, 3, 1, mean);
      out[date] = f;
    });
    return out;
  }

  // gauges: {dates: [...], chestatee_flow: [...], chattahoochee_flow: [...], lake_level: [...]}
  function gaugeFeatures(gauges) {
    const n = gauges.dates.length;
    const out = {};
    gauges.dates.forEach((d) => (out[d] = {}));
    for (const col of ["chestatee_flow", "chattahoochee_flow"]) {
      if (!gauges[col]) continue;
      const prev = [null, ...gauges[col].slice(0, n - 1)];
      gauges.dates.forEach((d, i) => {
        out[d][`${col}_log`] = isNum(prev[i]) ? Math.log10(Math.max(prev[i], 1)) : null;
        const med = windowStat(prev, i, 30, 7, median);
        out[d][`${col}_vs_30d`] = isNum(prev[i]) && isNum(med) ? prev[i] / med : null;
      });
    }
    if (gauges.lake_level) {
      const prev = [null, ...gauges.lake_level.slice(0, n - 1)];
      gauges.dates.forEach((d, i) => {
        out[d].lake_level = isNum(prev[i]) ? prev[i] : null;
        out[d].lake_level_change_7d = i >= 7 && isNum(prev[i]) && isNum(prev[i - 7]) ? prev[i] - prev[i - 7] : null;
      });
    }
    return out;
  }

  function calendarFeatures(date) {
    const doy = dayOfYear(date);
    const dow = new Date(date + "T00:00:00Z").getUTCDay(); // 0 = Sunday, 6 = Saturday
    return {
      season_sin: Math.sin((2 * Math.PI * doy) / 365.25),
      season_cos: Math.cos((2 * Math.PI * doy) / 365.25),
      weekend: dow === 0 || dow === 6 ? 1 : 0,
    };
  }

  function linear(parts, row, features) {
    let z = parts.intercept;
    features.forEach((name, j) => {
      const v = isNum(row[name]) ? row[name] : parts.fill[j];
      z += (parts.coef[j] * (v - parts.mean[j])) / parts.scale[j];
    });
    return z;
  }

  function predict(model, row) {
    const p = 1 / (1 + Math.exp(-linear(model.risk, row, model.features)));
    const q = {};
    for (const [k, parts] of Object.entries(model.quantiles)) q[k] = 10 ** linear(parts, row, model.features);
    const [q10, q50, q90] = [q["0.1"], q["0.5"], q["0.9"]].map((v) => Math.max(v, 1));
    let level = "low";
    if (q50 >= model.threshold || p >= model.cutoffs.avoid) level = "avoid";
    else if (q90 >= model.threshold || p >= model.cutoffs.caution) level = "caution";
    return { risk: p, range: [q10, Math.max(q50, q10), Math.max(q90, q50)], level };
  }

  // Build one row of model inputs for a site and day
  function buildRow(siteId, date, wf, gf, allSites) {
    const row = { ...(wf[date] || {}), ...(gf[date] || {}), ...calendarFeatures(date) };
    for (const s of allSites) row[`site_${s}`] = s === siteId ? 1 : 0;
    return row;
  }

  // Put USGS readings on a continuous daily calendar through lastDate, carrying the latest reading forward
  function dailyGauges(series, firstDate, lastDate) {
    const dates = [];
    for (let d = firstDate; d <= lastDate; d = addDays(d, 1)) dates.push(d);
    const out = { dates };
    for (const [name, pts] of Object.entries(series)) {
      const byDate = Object.fromEntries(pts.filter((p) => isNum(p.value)).map((p) => [p.date, p.value]));
      const known = Object.keys(byDate).sort();
      const lastKnown = known[known.length - 1];
      // Missing days in the past stay missing (like training); days after the latest reading reuse it
      out[name] = dates.map((d) => (d in byDate ? byDate[d] : d > lastKnown ? byDate[lastKnown] : null));
    }
    return out;
  }

  const api = { weatherFeatures, gaugeFeatures, calendarFeatures, predict, buildRow, dailyGauges, addDays };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LanierModel = api;
})(typeof window !== "undefined" ? window : globalThis);
