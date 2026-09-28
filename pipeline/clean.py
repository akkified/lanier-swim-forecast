"""Step 1: turn the Riverkeeper spreadsheet into one clean table.

Output: data/processed/samples.csv and data/processed/sites.json
"""
import json
import re

import numpy as np
import pandas as pd

from config import KEEP_DILUTION, PROCESSED, RAW_XLSX, THRESHOLD

COLUMNS = {
    "Site Name": "site_name",
    "Latitude": "lat",
    "Longitude": "lon",
    "Sample ID": "sample_id",
    "Collection Date": "date",
    "Collection Time": "time",
    "Total Coliform (MPN/100mL)": "total_coliform",
    "E. coli (MPN/100mL)": "ecoli",
    "Fluorometry": "fluorometry",
    "Turbidity (NTU)": "turbidity",
    "Conductivity (uS)": "conductivity",
    "Rainfall (in)": "rain_field",
    "#mL/100mL (Dilution)": "dilution",
    "Notes": "notes",
}


def site_id(name: str) -> str:
    name = name.replace("Lake Lanier at", "")
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def load_raw() -> pd.DataFrame:
    book = pd.ExcelFile(RAW_XLSX)
    frames = []
    for sheet in book.sheet_names:
        df = book.parse(sheet)
        if df.empty:
            continue
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMNS)[list(COLUMNS.values())]
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df[df["dilution"] == KEEP_DILUTION].copy()

    df["site_id"] = df["site_name"].map(site_id)
    df["ecoli"] = pd.to_numeric(df["ecoli"], errors="coerce")
    # A result of 1 is the lowest the test can report, so treat it as "below detection"
    df["below_detection"] = df["ecoli"] <= 1
    df["log_ecoli"] = np.log10(df["ecoli"].clip(lower=1))
    df["exceeds"] = (df["ecoli"] >= THRESHOLD).astype(int)

    df = df.drop_duplicates(subset=["site_id", "date", "sample_id"])
    return df.sort_values(["date", "site_id"]).reset_index(drop=True)


def main():
    df = clean(load_raw())
    df.to_csv(PROCESSED / "samples.csv", index=False)

    sites = (
        df.groupby("site_id")
        .agg(name=("site_name", "first"), lat=("lat", "mean"), lon=("lon", "mean"),
             samples=("ecoli", "size"), exceedances=("exceeds", "sum"))
        .reset_index()
    )
    sites["name"] = sites["name"].str.replace("Lake Lanier at ", "", regex=False)
    (PROCESSED / "sites.json").write_text(json.dumps(sites.to_dict("records"), indent=2))

    print(f"Kept {len(df)} samples at {df.site_id.nunique()} sites "
          f"({df.date.min().date()} to {df.date.max().date()})")
    print(f"Samples at or above {THRESHOLD}: {df.exceeds.sum()}")


if __name__ == "__main__":
    main()
