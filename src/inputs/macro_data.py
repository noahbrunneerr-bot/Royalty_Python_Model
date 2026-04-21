from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_FRED_SERIES = {
    "risk_free": "DGS10",        # 10Y Treasury
    "inflation": "CPIAUCSL",     # CPI All Urban Consumers
    "growth": "GDPC1",           # Real GDP
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_macro_csv_path() -> Path:
    return _project_root() / "Data" / "external" / "macro_sample.csv"


def _ensure_datetime_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col])
        out = out.sort_values(date_col).reset_index(drop=True)
    return out


def load_macro_series_from_csv(csv_path: str | Path | None = None) -> pd.DataFrame:
    """
    Load local fallback macro data from CSV.

    Expected columns:
    - date
    - risk_free
    - inflation
    - growth

    Additional columns are allowed.
    """
    path = Path(csv_path) if csv_path is not None else _default_macro_csv_path()
    if not path.exists():
        raise FileNotFoundError(f"Fallback macro CSV not found: {path}")

    df = pd.read_csv(path)
    required_cols = {"date", "risk_free", "inflation", "growth"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Fallback macro CSV is missing required columns: {sorted(missing)}")

    return _ensure_datetime_index(df, date_col="date")


def load_macro_series_from_fred(
    api_key: Optional[str] = None,
    csv_fallback_path: str | Path | None = None,
    start_date: str = "2010-01-01",
) -> pd.DataFrame:
    """
    Try to load macro series from FRED. Falls back to local CSV if:
    - no API key is supplied
    - fredapi is not installed
    - request/download fails

    Returns a DataFrame with columns:
    - date
    - risk_free
    - inflation
    - growth
    - source
    """
    if api_key is None or str(api_key).strip() == "":
        df = load_macro_series_from_csv(csv_fallback_path)
        df["source"] = "csv_fallback_no_api_key"
        return df

    try:
        from fredapi import Fred  # type: ignore
    except Exception:
        df = load_macro_series_from_csv(csv_fallback_path)
        df["source"] = "csv_fallback_no_fredapi"
        return df

    try:
        fred = Fred(api_key=api_key)

        rf = fred.get_series(DEFAULT_FRED_SERIES["risk_free"], observation_start=start_date)
        cpi = fred.get_series(DEFAULT_FRED_SERIES["inflation"], observation_start=start_date)
        gdp = fred.get_series(DEFAULT_FRED_SERIES["growth"], observation_start=start_date)

        # Daily 10Y Treasury -> monthly average
        df_rf = pd.DataFrame({"date": pd.to_datetime(rf.index), "risk_free": rf.values})
        df_rf = (
            df_rf.set_index("date")
            .resample("MS")
            .mean()
            .reset_index()
        )

        # CPI monthly level -> YoY inflation
        df_cpi = pd.DataFrame({"date": pd.to_datetime(cpi.index), "inflation_level": cpi.values})
        df_cpi = (
            df_cpi.set_index("date")
            .resample("MS")
            .last()
            .reset_index()
        )
        df_cpi["inflation"] = df_cpi["inflation_level"].pct_change(12) * 100.0

        # Real GDP quarterly level -> monthly forward-fill -> YoY growth
        df_gdp = pd.DataFrame({"date": pd.to_datetime(gdp.index), "growth_level": gdp.values})
        df_gdp = (
            df_gdp.set_index("date")
            .resample("MS")
            .ffill()
            .reset_index()
        )
        df_gdp["growth"] = df_gdp["growth_level"].pct_change(12) * 100.0

        df = (
            df_rf.merge(df_cpi[["date", "inflation"]], on="date", how="inner")
                 .merge(df_gdp[["date", "growth"]], on="date", how="inner")
        )

        df = df.dropna(subset=["risk_free", "inflation", "growth"]).reset_index(drop=True)
        df["source"] = "fred"

        if df.empty:
            raise ValueError("FRED download returned no usable rows after transformations.")

        return df

    except Exception:
        df = load_macro_series_from_csv(csv_fallback_path)
        df["source"] = "csv_fallback_request_failed"
        return df


def build_macro_scenarios(
    macro_df: pd.DataFrame,
    spread_base: float = 4.0,
    spread_high_rate: float = 5.0,
    spread_recession: float = 6.5,
    growth_shock_recession: float = -3.0,
    inflation_shock_high_rate: float = 1.0,
    use_recent_n: int = 12,
) -> pd.DataFrame:
    """
    Build a compact macro scenario table from historical macro inputs.

    Assumes input columns:
    - date
    - risk_free
    - inflation
    - growth

    Output columns:
    - Scenario
    - risk_free
    - inflation
    - growth
    - spread
    - discount_rate

    Notes:
    - all rates are returned in DECIMAL form for direct use in the valuation model
      e.g. 0.10 = 10%
    """
    df = macro_df.copy()

    required_cols = {"risk_free", "inflation", "growth"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"macro_df missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["risk_free", "inflation", "growth"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("macro_df has no usable rows after dropping NaNs.")

    recent = df.tail(use_recent_n)

    rf_base_pct = float(recent["risk_free"].mean())
    inflation_base_pct = float(recent["inflation"].mean())
    growth_base_pct = float(recent["growth"].mean())

    scenarios = [
        {
            "Scenario": "Base",
            "risk_free": rf_base_pct / 100.0,
            "inflation": inflation_base_pct / 100.0,
            "growth": growth_base_pct / 100.0,
            "spread": spread_base / 100.0,
        },
        {
            "Scenario": "High Rate",
            "risk_free": (rf_base_pct + max(0.75, inflation_shock_high_rate)) / 100.0,
            "inflation": (inflation_base_pct + inflation_shock_high_rate) / 100.0,
            "growth": max(growth_base_pct - 0.75, -2.0) / 100.0,
            "spread": spread_high_rate / 100.0,
        },
        {
            "Scenario": "Recession",
            "risk_free": max(rf_base_pct - 0.75, 1.0) / 100.0,
            "inflation": max(inflation_base_pct - 1.0, 0.5) / 100.0,
            "growth": min(growth_base_pct + growth_shock_recession, 0.5) / 100.0,
            "spread": spread_recession / 100.0,
        },
    ]

    out = pd.DataFrame(scenarios)
    out["discount_rate"] = out["risk_free"] + out["spread"]

    return out


def save_macro_scenarios(
    macro_scenarios: pd.DataFrame,
    output_path: str | Path | None = None,
) -> Path:
    """
    Save scenario table to CSV.
    """
    if output_path is None:
        output_path = _project_root() / "outputs" / "macro_scenarios.csv"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    macro_scenarios.to_csv(output_path, index=False)
    return output_path