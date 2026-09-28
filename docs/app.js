// Lanier Swim Forecast front end. Reads data/forecast.json made by pipeline/forecast.py.

const LEVEL_TEXT = { low: "Low risk", caution: "Caution", avoid: "Avoid swimming" };
const LEVEL_COLOR = () => {
  const css = getComputedStyle(document.documentElement);
  return {
    low: css.getPropertyValue("--low").trim(),
    caution: css.getPropertyValue("--caution").trim(),
    avoid: css.getPropertyValue("--avoid").trim(),
  };
};

let data;
let dayIndex = 0;
let selected = null;
let map;
const markers = {};

function dayLabel(iso, i) {
  if (i === 0) return "Today";
  const d = new Date(iso + "T12:00:00");
  return d.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
}

function fmtDate(iso) {
  return new Date(iso + "T12:00:00").toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function renderTabs() {
  const tabs = document.getElementById("day-tabs");
  tabs.innerHTML = "";
  data.sites[0].days.forEach((d, i) => {
    const b = document.createElement("button");
    b.textContent = dayLabel(d.date, i);
    b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", i === dayIndex);
    b.onclick = () => { dayIndex = i; render(); };
    tabs.appendChild(b);
  });
}

function renderList() {
  const list = document.getElementById("beach-list");
  list.innerHTML = "";
  const order = { avoid: 0, caution: 1, low: 2 };
  [...data.sites]
    .sort((a, b) => order[a.days[dayIndex].level] - order[b.days[dayIndex].level] || a.name.localeCompare(b.name))
    .forEach((s) => {
      const d = s.days[dayIndex];
      const li = document.createElement("li");
      li.className = selected === s.id ? "active" : "";
      li.innerHTML = `<span>${s.name}</span><span class="pill ${d.level}">${LEVEL_TEXT[d.level]}</span>`;
      li.onclick = () => select(s.id);
      list.appendChild(li);
    });
}

function renderMarkers() {
  if (!map) return;
  const colors = LEVEL_COLOR();
  data.sites.forEach((s) => {
    const d = s.days[dayIndex];
    const style = {
      radius: selected === s.id ? 12 : 9,
      color: "#fff",
      weight: 2,
      fillColor: colors[d.level],
      fillOpacity: 0.95,
    };
    if (!markers[s.id]) {
      markers[s.id] = L.circleMarker([s.lat, s.lon], style).addTo(map);
      markers[s.id].on("click", () => select(s.id));
    } else {
      markers[s.id].setStyle(style);
    }
    markers[s.id].bindTooltip(`${s.name}: ${LEVEL_TEXT[d.level]}`);
  });
}

function renderDetail() {
  const box = document.getElementById("detail");
  const s = data.sites.find((x) => x.id === selected);
  if (!s) { box.hidden = true; return; }
  const d = s.days[dayIndex];
  const [lo, mid, hi] = d.range.map((v) => Math.round(v));
  const rain = (v) => (v === null ? "n/a" : `${v.toFixed(2)} in`);
  box.hidden = false;
  box.innerHTML = `
    <h2>${s.name}</h2>
    <span class="pill ${d.level}">${LEVEL_TEXT[d.level]}</span>
    <span class="fine"> for ${dayLabel(d.date, dayIndex)}</span>
    <div class="stats">
      <div class="stat"><div class="k">Likely E. coli range</div><div class="v">${lo} to ${hi}</div><div class="fine">Best guess ${mid} per 100 mL</div></div>
      <div class="stat"><div class="k">Rain that day</div><div class="v">${rain(d.rain_today_in)}</div></div>
      <div class="stat"><div class="k">Rain in the 3 days before</div><div class="v">${rain(d.rain_prev_3d_in)}</div></div>
      <div class="stat"><div class="k">Last Riverkeeper test</div><div class="v">${Math.round(s.last_sample.ecoli)}</div><div class="fine">${fmtDate(s.last_sample.date)}</div></div>
      <div class="stat"><div class="k">Past samples over the limit</div><div class="v">${s.exceedances} of ${s.samples}</div></div>
    </div>`;
}

function renderAbout() {
  const total = data.sites.reduce((n, s) => n + s.samples, 0);
  document.getElementById("n-samples").textContent = total;
  document.getElementById("threshold").textContent = data.threshold;
  const v = data.validation || {};
  const gb = v.primary_scores;
  document.getElementById("validation").textContent = gb
    ? `Tested on ${v.tested_samples} past samples, one season at a time, without the model seeing that season first. ` +
      `Of the ${gb.total_high} samples that went over the limit, ${gb.caught_in_top10pct} landed in the model's riskiest 10% of days. ` +
      `Ranking score (AUC): ${gb.auc} (0.5 is a coin flip, 1.0 is perfect). ` +
      `The likely range held the real result ${Math.round(v.range_model.share_inside_10_90_range * 100)}% of the time.`
    : "Test results will show here after the model is trained.";
  const when = new Date(data.generated).toLocaleString();
  const src = data.source === "live" ? "Live forecast made in your browser" : "Saved forecast";
  let note = `${src}, ${when}.`;
  if (data.missing && data.missing.length) note += ` Some river or lake readings were unavailable (${data.missing.join(", ")}), so the model used typical values for them.`;
  document.getElementById("generated").textContent = note;

  const banner = document.getElementById("demo-banner");
  const month = new Date().getMonth() + 1;
  if (data.demo || data.source === "demo") {
    banner.textContent = "Demo mode: this forecast uses made-up weather for testing. Do not use it to decide whether to swim.";
    banner.hidden = false;
  } else if (month < 5 || month > 9) {
    banner.textContent = "Swim season is over. The model was trained on summer samples, so forecasts outside May to September are less reliable.";
    banner.hidden = false;
  } else {
    banner.hidden = true;
  }
}

function select(id) {
  selected = id;
  const s = data.sites.find((x) => x.id === id);
  if (map) map.panTo([s.lat, s.lon]);
  render();
  document.getElementById("detail").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function render() {
  renderTabs();
  renderList();
  renderMarkers();
  renderDetail();
}

// ---------- Live data ----------

const PAST_DAYS = 40;
const FORECAST_DAYS = 3;
const GAUGES = {
  chestatee_flow: ["02333500", "00060"],
  chattahoochee_flow: ["02331600", "00060"],
  lake_level: ["02334400", "00062"],
};

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} from ${new URL(url).host}`);
  return res.json();
}

async function liveWeather(sites) {
  const url = "https://api.open-meteo.com/v1/forecast?" + new URLSearchParams({
    latitude: sites.map((s) => s.lat).join(","),
    longitude: sites.map((s) => s.lon).join(","),
    daily: "precipitation_sum,temperature_2m_max",
    timezone: "America/New_York",
    past_days: PAST_DAYS,
    forecast_days: FORECAST_DAYS,
  });
  let res = await getJSON(url);
  if (!Array.isArray(res)) res = [res];
  return res.map((r) => ({ dates: r.daily.time, rain_mm: r.daily.precipitation_sum, temp_max_c: r.daily.temperature_2m_max }));
}

// USGS daily values: try the newer Water Data API, then the older WaterServices API
async function liveGauge(site, param, start, end) {
  try {
    const url = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items?" + new URLSearchParams({
      monitoring_location_id: `USGS-${site}`, parameter_code: param, statistic_id: "00003",
      datetime: `${start}/${end}`, f: "json", limit: 1000,
    });
    const j = await getJSON(url);
    const pts = (j.features || []).map((f) => ({ date: f.properties.time.slice(0, 10), value: +f.properties.value }));
    if (pts.length) return pts.filter((p) => p.value >= 0);
  } catch (e) { /* fall through */ }
  const url = "https://waterservices.usgs.gov/nwis/dv/?" + new URLSearchParams({
    format: "json", sites: site, parameterCd: param, startDT: start, endDT: end,
  });
  const j = await getJSON(url);
  const ts = j.value.timeSeries;
  if (!ts.length) return [];
  return ts[0].values[0].value.map((v) => ({ date: v.dateTime.slice(0, 10), value: +v.value })).filter((p) => p.value >= 0);
}

async function buildLiveForecast(model) {
  const M = window.LanierModel;
  const sites = model.sites;
  const weather = await liveWeather(sites);
  const dates = weather[0].dates;
  const today = dates[PAST_DAYS];
  const first = dates[0];
  const last = dates[dates.length - 1];

  const series = {};
  const missing = [];
  await Promise.all(Object.entries(GAUGES).map(async ([name, [site, param]]) => {
    try {
      const pts = await liveGauge(site, param, first, today);
      if (pts.length) series[name] = pts; else missing.push(name);
    } catch (e) { missing.push(name); }
  }));
  const gf = M.gaugeFeatures(M.dailyGauges(series, first, last));
  const siteIds = sites.map((s) => s.site_id);

  return {
    generated: new Date().toISOString(),
    source: "live",
    missing,
    threshold: model.threshold,
    validation: model.validation,
    sites: sites.map((s, k) => {
      const wf = M.weatherFeatures(weather[k]);
      const days = dates.slice(PAST_DAYS).map((date) => {
        const row = M.buildRow(s.site_id, date, wf, gf, siteIds);
        const r = M.predict(model, row);
        return {
          date, risk: r.risk, level: r.level, range: r.range,
          rain_today_in: row.rain_today == null ? null : row.rain_today / 25.4,
          rain_prev_3d_in: row.rain_prev_3d == null ? null : row.rain_prev_3d / 25.4,
        };
      });
      return { id: s.site_id, name: s.name, lat: s.lat, lon: s.lon, samples: s.samples,
               exceedances: s.exceedances, last_sample: s.last_sample, days };
    }),
  };
}

async function init() {
  if (window.L) {
    map = L.map("map", { scrollWheelZoom: false });
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 16,
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(map);
  } else {
    document.querySelector(".map-card").hidden = true; // map library did not load; the list still works
  }

  try {
    const model = await (await fetch("data/model.json", { cache: "no-store" })).json();
    data = await buildLiveForecast(model);
  } catch (e) {
    console.warn("Live forecast failed, using saved forecast:", e);
    try {
      data = await (await fetch("data/forecast.json", { cache: "no-store" })).json();
      data.source = data.demo ? "demo" : "saved";
    } catch (e2) {
      document.getElementById("beach-list").innerHTML =
        "<li>Could not load the forecast. Serve this folder with python3 -m http.server and check your internet connection.</li>";
      return;
    }
  }
  if (map) map.fitBounds(data.sites.map((s) => [s.lat, s.lon]), { padding: [30, 30] });
  renderAbout();
  render();
}

init();
