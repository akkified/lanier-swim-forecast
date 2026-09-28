"""Shared settings for the Lake Lanier E. coli forecast."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_XLSX = ROOT / "data" / "raw" / "lanier_swim_guide.xlsx"
PROCESSED = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
WEB_DATA = ROOT / "docs" / "data"

# EPA single-sample swim advisory level used by Swim Guide (MPN per 100 mL)
THRESHOLD = 235

# Riverkeeper switched Lanier samples from a 2 mL to a 100 mL dilution.
# Their monitoring manager suggested dropping the older 2 mL samples.
KEEP_DILUTION = 100

# Lake-wide inputs from USGS. These are optional: if a download fails,
# the model trains without them.
USGS_GAUGES = {
    "chestatee_flow": ("02333500", "00060"),   # Chestatee River near Dahlonega, discharge (cfs)
    "chattahoochee_flow": ("02331600", "00060"),  # Chattahoochee River near Cornelia/Leaf, discharge (cfs)
    "lake_level": ("02334400", "62614"),       # Lake Sidney Lanier near Buford, elevation (ft)
}

# Days of rain history to use as features
RAIN_WINDOWS = [1, 2, 3, 7]

# How many days ahead the app forecasts (today plus this many)
FORECAST_DAYS = 3

for p in (PROCESSED, CACHE, MODELS, REPORTS, WEB_DATA):
    p.mkdir(parents=True, exist_ok=True)
