from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_FRED_SERIES = {
    "risk_free": "DGS10",        # 10Y Treasury
    "inflation": "CPIAUCSL",     # CPI All Urban Consumers
    "growth": "GDPC1",           # Real GDP
    "fed_funds": "FEDFUNDS",     # Effective Federal Funds Rate
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


def get_fred_api_key(explicit_api_key: Optional[str] = None) -> Optional[str]:
    """
    Resolve a FRED API key in this order:
    1) explicit function argument
    2) Streamlit secrets (if available)
    3) environment variable FRED_API_KEY

    Returns None if no key is available.
    """
    if explicit_api_key is not None and str(explicit_api_key).strip():
        return str(explicit_api_key).strip()

    # Try Streamlit secrets
    try:
        import streamlit as st  # type: ignore

        if hasattr(st, "secrets") and "FRED_API_KEY" in st.secrets:
            key = str(st.secrets["FRED_API_KEY"]).strip()
            if key:
                return key
    except Exception:
        pass

    # Try environment variable
    env_key = os.getenv("FRED_API_KEY")
    if env_key and str(env_key).strip():
        return str(env_key).strip()

    return None


def load_macro_series_from_csv(csv_path: str | Path | None = None) -> pd.DataFrame:
    """
    Load local fallback macro data from CSV.

    Expected columns:
    - date
    - risk_free
    - inflation
    - growth

    Optional columns:
    - fed_funds

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

    df = _ensure_datetime_index(df, date_col="date")

    # Ensure optional fed_funds column exists for downstream logic
    if "fed_funds" not in df.columns:
        df["fed_funds"] = pd.NA

    df["source"] = "csv_fallback"
    return df


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
    - fed_funds
    - source
    """
    resolved_api_key = get_fred_api_key(api_key)

    if resolved_api_key is None:
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
        fred = Fred(api_key=resolved_api_key)

        rf = fred.get_series(DEFAULT_FRED_SERIES["risk_free"], observation_start=start_date)
        cpi = fred.get_series(DEFAULT_FRED_SERIES["inflation"], observation_start=start_date)
        gdp = fred.get_series(DEFAULT_FRED_SERIES["growth"], observation_start=start_date)
        fed_funds = fred.get_series(DEFAULT_FRED_SERIES["fed_funds"], observation_start=start_date)

        # 10Y Treasury daily -> monthly average
        df_rf = pd.DataFrame({"date": pd.to_datetime(rf.index), "risk_free": pd.to_numeric(rf.values, errors="coerce")})
        df_rf = (
            df_rf.set_index("date")
            .resample("MS")
            .mean()
            .reset_index()
        )

        # CPI monthly level -> YoY inflation %
        df_cpi = pd.DataFrame({"date": pd.to_datetime(cpi.index), "inflation_level": pd.to_numeric(cpi.values, errors="coerce")})
        df_cpi = (
            df_cpi.set_index("date")
            .resample("MS")
            .last()
            .reset_index()
        )
        df_cpi["inflation"] = df_cpi["inflation_level"].pct_change(12) * 100.0

        # GDP quarterly level -> monthly forward fill -> YoY real growth %
        df_gdp = pd.DataFrame({"date": pd.to_datetime(gdp.index), "growth_level": pd.to_numeric(gdp.values, errors="coerce")})
        df_gdp = (
            df_gdp.set_index("date")
            .resample("MS")
            .ffill()
            .reset_index()
        )
        df_gdp["growth"] = df_gdp["growth_level"].pct_change(12) * 100.0

        # Fed funds monthly average
        df_ff = pd.DataFrame({"date": pd.to_datetime(fed_funds.index), "fed_funds": pd.to_numeric(fed_funds.values, errors="coerce")})
        df_ff = (
            df_ff.set_index("date")
            .resample("MS")
            .mean()
            .reset_index()
        )

        df = (
            df_rf.merge(df_cpi[["date", "inflation"]], on="date", how="inner")
                 .merge(df_gdp[["date", "growth"]], on="date", how="inner")
                 .merge(df_ff[["date", "fed_funds"]], on="date", how="left")
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


def get_macro_data(
    use_fred: bool = True,
    api_key: Optional[str] = None,
    csv_fallback_path: str | Path | None = None,
    start_date: str = "2010-01-01",
) -> tuple[pd.DataFrame, str]:
    """
    Main wrapper for macro data loading.

    Returns:
    - macro_df
    - macro_source string
    """
    if use_fred:
        df = load_macro_series_from_fred(
            api_key=api_key,
            csv_fallback_path=csv_fallback_path,
            start_date=start_date,
        )
    else:
        df = load_macro_series_from_csv(csv_fallback_path)
        df["source"] = "csv_fallback_forced"

    source = str(df["source"].iloc[-1]) if "source" in df.columns and not df.empty else "unknown"
    return df, source


def infer_macro_regime(latest_row: pd.Series | dict) -> str:
    """
    Infer a simple macro regime from latest macro observations.

    Uses:
    - risk_free (10Y Treasury proxy)
    - fed_funds (if available)

    Regimes:
    - Benign
    - Base
    - High Rate
    - Stress
    """
    if isinstance(latest_row, dict):
        risk_free = latest_row.get("risk_free")
        fed_funds = latest_row.get("fed_funds")
    else:
        risk_free = latest_row.get("risk_free", None)
        fed_funds = latest_row.get("fed_funds", None)

    try:
        rf = float(risk_free) if pd.notna(risk_free) else None
    except Exception:
        rf = None

    try:
        ff = float(fed_funds) if pd.notna(fed_funds) else None
    except Exception:
        ff = None

    if rf is None and ff is None:
        return "Base"

    # Conservative default if only one series exists
    metric = max([x for x in [rf, ff] if x is not None], default=3.5)

    if metric < 3.0:
        return "Benign"
    if metric < 4.5:
        return "Base"
    if metric < 5.5:
        return "High Rate"
    return "Stress"


def macro_discount_adjustment(regime: str) -> float:
    """
    Return a simple discount-rate overlay in DECIMAL form.

    Example:
    0.010 = +1.0%
    """
    regime_clean = str(regime).strip().lower()

    if regime_clean == "benign":
        return -0.005
    if regime_clean == "base":
        return 0.000
    if regime_clean == "high rate":
        return 0.010
    if regime_clean == "stress":
        return 0.015
    return 0.000


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

    Optional:
    - fed_funds
    - source

    Output columns:
    - Scenario
    - risk_free
    - inflation
    - growth
    - spread
    - discount_rate
    - macro_regime
    - source

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

    latest_row = recent.iloc[-1]
    macro_regime = infer_macro_regime(latest_row)
    source = str(latest_row["source"]) if "source" in latest_row.index else "unknown"

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
    out["macro_regime"] = macro_regime
    out["source"] = source

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