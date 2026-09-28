# Lanier Swim Forecast

A web app that estimates E. coli risk at Lake Lanier beaches for today and the next two days,
built for the 2026 Congressional App Challenge.

Chattahoochee Riverkeeper tests seven Lanier beaches once a week in the summer. This project
trains a machine learning model on those samples plus weather, river flow, and lake level so the
app can estimate risk on the days in between.

## Folder layout

```
cac/
  data/raw/lanier_swim_guide.xlsx   Riverkeeper sample data (do not edit)
  data/raw/lanier_inputs.json       daily weather (Open-Meteo) and USGS river/lake data, Jan 2019 to Sept 2026
  data/processed/                   cleaned samples and site list (made by clean.py)
  data/cache/                       downloaded weather and USGS data
  pipeline/
    config.py      settings (swim limit, USGS gauges, forecast length)
    clean.py       step 1: clean the spreadsheet
    weather.py     rainfall and temperature from Open-Meteo
    usgs.py        river flow and lake level from USGS
    features.py    turns raw data into model inputs
    train.py       step 2: train and test the models
    local_inputs.py  loads data/raw/lanier_inputs.json so training works offline
    forecast.py    step 3 (optional): write a saved forecast to docs/data/forecast.json
  models/          saved model
  reports/         test results (metrics.json, cv_predictions.csv)
  docs/            the website (GitHub Pages serves this folder)
    index.html, style.css
    app.js         page logic; downloads live weather + USGS data and runs the model
    model.js       the model in JavaScript (same math as the Python code)
    data/model.json  trained model exported as numbers
    data/forecast.json  backup forecast shown if live data can't be reached
  tests/check_browser_model.py   checks model.js gives the same answers as Python
```

Live site: https://akkified.github.io/lanier-swim-forecast/

## How the website works

When someone opens the page, their browser downloads the latest weather forecast from Open-Meteo
and the last 40 days of river flow and lake level from USGS, builds the model inputs, and runs the
model right there. No server is needed, so the site can be hosted free on GitHub Pages.
If live data can't be reached, it shows the saved forecast instead.

## Setup (one time)

```bash
cd ~/cac
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run it

```bash
cd ~/cac/pipeline
python clean.py        # clean the Riverkeeper data
python train.py        # train, test, and export docs/data/model.json
cd ..
python tests/check_browser_model.py   # confirm the browser model matches

cd docs
python -m http.server 8000
# open http://localhost:8000
```

`train.py` uses `data/raw/lanier_inputs.json` if it exists. Delete that file to have it download
fresh data instead. `python train.py --offline` trains with only the spreadsheet's rainfall column.

## How the model is tested

Leave-one-season-out: train on every summer except one, predict the held-out summer, and repeat
for each year. The model never sees any data from the season it is graded on. Results are saved
to `reports/metrics.json`:

- `p_base`: a simple baseline that only uses how often each beach has gone over the limit before
- `p_lr`: logistic regression
- `p_gb`: gradient boosted trees
- whichever of those two scores better becomes the main model
- `auc`: 0.5 is a coin flip, 1.0 is perfect
- `caught_in_top10pct`: how many of the real high samples landed in the model's riskiest 10% of days
- `range_model`: how often the real result fell inside the predicted 10th to 90th percentile range

The main model has to beat the baseline to be worth using.

### Results (September 2026)

| Model | AUC | High samples caught in riskiest 10% of days |
|---|---|---|
| Baseline (beach history only) | 0.44 | 0 of 16 |
| Gradient boosting | 0.46 | 2 of 16 |
| Logistic regression (main) | 0.79 | 5 of 16 |

- AUC 95% bootstrap interval for the main model: 0.68 to 0.88
- Shuffle test: a score this high happened by chance in about 4 of 1,000 shuffles (p = 0.004)
- The likely range held the real result 74% of the time
- Biggest drivers of risk: rain over the past week, warm days, high Chestatee River flow, high lake level,
  and the Buford Dam beach. Rain on the sampling day alone did not predict high results.
- Gradient boosting did worse because 16 high samples is too few for a complex model.

## Data notes

- The Riverkeeper sample spreadsheet is not included in this repository. It was shared by
  Chattahoochee Riverkeeper for this project; contact them for access. To retrain, place it at
  `data/raw/lanier_swim_guide.xlsx`.
- Only samples run at the 100 mL dilution are used, per Riverkeeper's Headwaters Water Monitoring Manager.
- A result of 1 is treated as below the detection limit.
- The swim limit is 235 MPN per 100 mL (EPA single sample level used by Swim Guide).
- 302 usable samples, 16 over the limit. High days are rare, so the model predicts a chance of
  going over rather than an exact number.

## Credits

Sample data: Chattahoochee Riverkeeper. Weather: Open-Meteo. River and lake data: USGS.
Map: Leaflet and OpenStreetMap.
