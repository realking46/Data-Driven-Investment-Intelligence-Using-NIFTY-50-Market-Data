from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


PRICE_COLUMN_ALIASES = {
    "date": "date",
    "symbol": "symbol",
    "series": "series",
    "prev close": "prev_close",
    "prev_close": "prev_close",
    "open": "open",
    "high": "high",
    "low": "low",
    "last": "last",
    "close": "close",
    "vwap": "vwap",
    "volume": "volume",
    "turnover": "turnover",
    "trades": "trades",
    "deliverable volume": "deliverable_volume",
    "%deliverble": "deliverable_percent",
    "%deliverable": "deliverable_percent",
    "deliverble_percent": "deliverable_percent",
    "deliverable_percent": "deliverable_percent",
}

REQUIRED_PRICE_COLUMNS = {"date", "open", "high", "low", "close", "volume"}


def list_csv_files(raw_dir: str | Path) -> list[Path]:
    """Return CSV files in the raw-data directory."""
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return []
    return sorted(raw_path.rglob("*.csv"))


def normalize_column_name(name: str) -> str:
    name = str(name).strip().lower()
    name = name.replace("\ufeff", "")
    name = name.replace("_", " ")
    name = " ".join(name.split())
    return PRICE_COLUMN_ALIASES.get(name, name.replace(" ", "_"))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_column_name(col) for col in out.columns]
    return out


def looks_like_price_file(columns: Iterable[str]) -> bool:
    normalized = {normalize_column_name(col) for col in columns}
    return REQUIRED_PRICE_COLUMNS.issubset(normalized)


def looks_like_metadata_file(columns: Iterable[str]) -> bool:
    normalized = {normalize_column_name(col) for col in columns}
    return bool({"symbol", "company_name", "company", "sector", "industry"} & normalized) and "date" not in normalized


def _to_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == "object":
        series = series.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(series, errors="coerce")


def read_price_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = normalize_columns(pd.read_csv(path))

    if not looks_like_price_file(df.columns):
        raise ValueError(f"{path.name} does not look like a price-history CSV.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_columns = [
        "prev_close",
        "open",
        "high",
        "low",
        "last",
        "close",
        "vwap",
        "volume",
        "turnover",
        "trades",
        "deliverable_volume",
        "deliverable_percent",
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = _to_numeric(df[col])

    if "symbol" not in df.columns:
        df["symbol"] = path.stem.upper()

    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["source_file"] = path.name
    return df


def load_price_history(raw_dir: str | Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for csv_path in list_csv_files(raw_dir):
        try:
            preview = pd.read_csv(csv_path, nrows=5)
        except Exception:
            continue

        if not looks_like_price_file(preview.columns):
            continue

        frames.append(read_price_file(csv_path))

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["symbol", "date"]).reset_index(drop=True)


def load_metadata(raw_dir: str | Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for csv_path in list_csv_files(raw_dir):
        try:
            preview = pd.read_csv(csv_path, nrows=5)
        except Exception:
            continue

        if not looks_like_metadata_file(preview.columns):
            continue

        df = normalize_columns(pd.read_csv(csv_path))
        df["source_file"] = csv_path.name
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)

