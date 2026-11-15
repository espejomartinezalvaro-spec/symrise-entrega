"""Data loading and cleaning shared by all notebooks.

- load_raw(): reads and validates the raw file; does NOT clean anything.
- clean():    applies the cleaning decisions DQ-01..DQ-08, each shown with its
              rationale in notebooks/03_data_preparation.ipynb.
- load_clean(): reads the processed parquet written by the phase 3 notebook.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

# Expected raw schema (as delivered, which differs from the use case statement)
RAW_DTYPES = {
    "batch_id": "string",
    "event_timestamp": "string",  # mixed formats -> parsed in clean()
    "plant_location": "category",
    "plant_country": "category",
    "production_line_id": "category",
    "temp_value": "float64",
    "temp_unit": "category",
    "vessel_pressure_bar": "float64",
    "refractive_index": "float64",
    "density_g_cm3": "float64",
    "ph_level": "float64",
    "mixing_torque_nm": "float64",
    "mass_flow_rate_kg_h": "float64",
    "stability_months": "float64",
    "purity_grade": "category",
    "oxidation_risk_flag": "boolean",
}

# Sensor columns after cleaning (temperature renamed to the documented name)
SENSORS = [
    "reaction_temperature_celsius",
    "vessel_pressure_bar",
    "refractive_index",
    "density_g_cm3",
    "ph_level",
    "mixing_torque_nm",
    "mass_flow_rate_kg_h",
]

# DQ-03: one exact, physically impossible value per sensor written by failing sensors
ERROR_CODES = {
    "reaction_temperature_celsius": 210.0,  # 210 °C (also appears as 410 °F)
    "vessel_pressure_bar": -1.0,
    "density_g_cm3": 860.0,
    "ph_level": 14.0,
}

PROCESSED_FILE = "fragrance_clean.parquet"


def load_config() -> dict:
    with open(ROOT / "configs" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw() -> pd.DataFrame:
    """Read the raw CSV with explicit dtypes and check the schema."""
    cfg = load_config()
    df = pd.read_csv(ROOT / cfg["paths"]["raw_data"], dtype=RAW_DTYPES)

    assert list(df.columns) == list(RAW_DTYPES), "Unexpected columns in raw file"
    assert df["batch_id"].is_unique, "batch_id is not unique"
    return df


def parse_dates(ts: pd.Series) -> pd.Series:
    """DQ-01: parse the 3 timestamp formats explicitly and keep the day.

    Slash dates are dd/mm/yyyy (the month field only takes values 4-6).
    Slash and text formats have no time of day, so the day is the common grain.
    """
    iso = pd.to_datetime(ts, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    slash = pd.to_datetime(ts, format="%d/%m/%Y", errors="coerce")
    text = pd.to_datetime(ts, format="%b %d, %Y", errors="coerce")
    return iso.fillna(slash).fillna(text).dt.floor("D")


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning decisions DQ-01..DQ-08 to the raw data."""
    df = raw.copy()

    # DQ-01: one production date column (daily grain)
    df["date"] = parse_dates(df["event_timestamp"])

    # DQ-02: Fahrenheit -> Celsius; one temperature column with the documented name
    is_f = df["temp_unit"] == "F"
    df["reaction_temperature_celsius"] = np.where(
        is_f, (df["temp_value"] - 32) * 5 / 9, df["temp_value"]
    )

    # DQ-03: error codes are not measurements -> set to missing (keep a flag)
    is_code = pd.DataFrame({c: np.isclose(df[c], v) for c, v in ERROR_CODES.items()})
    df["had_error_code"] = is_code.any(axis=1)
    for c in ERROR_CODES:
        df.loc[is_code[c], c] = np.nan

    # DQ-03b: marginal excursions are kept as real readings (no action)
    # DQ-04: remaining missing values are kept as NaN (handled in each analysis/model)

    # DQ-05: drop batches with no sensor information at all
    df = df[df[SENSORS].notna().any(axis=1)]

    # DQ-08: physical reactor = plant x reactor
    plant = df["plant_location"].astype(str).str[-2:]
    reactor = df["production_line_id"].astype(str).str[-2:]
    df["reactor_id"] = ("P" + plant + "-R" + reactor).astype("category")

    # DQ-06: drop redundant / replaced columns. DQ-07: leakage columns are kept
    # for business reporting but listed in config.yaml so they are never features.
    df = df.rename(columns={"plant_location": "plant"})
    columns = (
        ["batch_id", "date", "plant", "reactor_id"]
        + SENSORS
        + ["had_error_code", "stability_months", "purity_grade", "oxidation_risk_flag"]
    )
    df = df[columns].reset_index(drop=True)

    # Checks on the output
    assert df["date"].notna().all(), "Unparsed dates"
    assert df["batch_id"].is_unique
    for c, v in ERROR_CODES.items():
        assert not np.isclose(df[c], v).any(), f"Error code left in {c}"
    assert df["reaction_temperature_celsius"].max() < 100, "Temperature not in °C"
    assert df[SENSORS].notna().any(axis=1).all(), "Rows without any sensor"
    assert df["stability_months"].between(18, 30).all(), "Target out of range"
    return df


def save_clean(df: pd.DataFrame) -> Path:
    path = ROOT / load_config()["paths"]["processed_dir"] / PROCESSED_FILE
    df.to_parquet(path, index=False)
    return path


def load_clean() -> pd.DataFrame:
    return pd.read_parquet(ROOT / load_config()["paths"]["processed_dir"] / PROCESSED_FILE)
