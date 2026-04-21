from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.pir_waterfall_engine import (
    WaterfallParams,
    ScenarioSeries,
    compute_waterfall,
    compute_pg_equity_cashflows,
)
from src.metrics import irr_annual, moic


def _safe_rel_diff(python_value: float, excel_value: float) -> float:
    if pd.isna(excel_value) or excel_value == 0:
        return np.nan
    return (python_value - excel_value) / excel_value


def load_ground_truth(csv_path: str | Path) -> pd.DataFrame:
    """
    Load cleaned PG3 ground truth CSV.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    required_cols = [
        "Scenario",
        "FY",
        "Net_CF_to_Consortium",
        "Consortium_Fees",
        "Interest_Rate",
        "Mandatory_Amort",
        "NAV_Multiple",
        "Debt_End",
        "FCF_for_Distribution",
        "PG_Share",
        "Equity_Ticket",
        "Equity_CF",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in ground truth: {missing}")

    return df.copy()


def build_params_from_ground_truth(
    gt_df: pd.DataFrame,
    *,
    scenario: str = "Base",
    recap_target_ltv: float = 0.60,
    recap_years: Tuple[int, ...] = (2026, 2029),
    cash_sweep_start_fy: int = 2027,
    operating_fee_pct: float = 0.05,
    apply_cash_sweep_only_for: Tuple[str, ...] = ("Flat", "Downside"),
    terminal_override: bool = True,
) -> WaterfallParams:
    """
    Build current Python engine params from ground truth anchors.
    """
    base = gt_df.loc[gt_df["Scenario"].astype(str).str.strip() == scenario].copy()
    if base.empty:
        raise ValueError(f"No rows found for scenario '{scenario}'")

    return WaterfallParams(
        scenario=scenario,
        entry_debt=float(base["Debt_End"].iloc[0]),
        recap_target_ltv=recap_target_ltv,
        recap_years=recap_years,
        cash_sweep_start_fy=cash_sweep_start_fy,
        operating_fee_pct=operating_fee_pct,
        pg_share=float(base["PG_Share"].iloc[0]),
        consortium_equity_ticket=float(base["Equity_Ticket"].iloc[0]),
        apply_cash_sweep_only_for=apply_cash_sweep_only_for,
        terminal_override=terminal_override,
    )


def build_series_from_ground_truth(
    gt_df: pd.DataFrame,
    scenario: str = "Base",
) -> ScenarioSeries:
    """
    Extract ScenarioSeries for the selected scenario from ground truth.
    """
    base = gt_df.loc[gt_df["Scenario"].astype(str).str.strip() == scenario].copy()
    if base.empty:
        raise ValueError(f"No rows found for scenario '{scenario}'")

    return ScenarioSeries(
        fy=base["FY"].to_numpy(dtype=int),
        net_cf_to_consortium=base["Net_CF_to_Consortium"].to_numpy(dtype=float),
        interest_rate=base["Interest_Rate"].to_numpy(dtype=float),
        nav_multiple=base["NAV_Multiple"].to_numpy(dtype=float),
        mandatory_amort=base["Mandatory_Amort"].to_numpy(dtype=float),
        consortium_fees=base["Consortium_Fees"].to_numpy(dtype=float),
    )


def build_yearly_comparison(
    gt_df: pd.DataFrame,
    scenario: str = "Base",
    **param_overrides,
) -> pd.DataFrame:
    """
    Compare Excel/ground-truth yearly outputs with current Python waterfall logic.
    """
    base = gt_df.loc[gt_df["Scenario"].astype(str).str.strip() == scenario].copy()
    params = build_params_from_ground_truth(gt_df, scenario=scenario, **param_overrides)
    series = build_series_from_ground_truth(gt_df, scenario=scenario)

    py_wf = compute_waterfall(params, series)
    py_eq = compute_pg_equity_cashflows(params, py_wf)

    compare = pd.DataFrame(
        {
            "Scenario": base["Scenario"].to_numpy(),
            "FY": base["FY"].to_numpy(),
            "gt_Net_CF_to_Consortium": base["Net_CF_to_Consortium"].to_numpy(dtype=float),
            "py_Net_CF_to_Consortium": py_wf["Net CF to Consortium"].to_numpy(dtype=float),
            "gt_Consortium_Fees": base["Consortium_Fees"].to_numpy(dtype=float),
            "py_Consortium_Fees": py_wf["Consortium Fees"].to_numpy(dtype=float),
            "gt_Interest_Rate": base["Interest_Rate"].to_numpy(dtype=float),
            "py_Interest_Rate": py_wf["Interest Rate"].to_numpy(dtype=float),
            "gt_Mandatory_Amort": base["Mandatory_Amort"].to_numpy(dtype=float),
            "py_Mandatory_Amort": py_wf["Mandatory Amort"].to_numpy(dtype=float),
            "gt_Debt_End": base["Debt_End"].to_numpy(dtype=float),
            "py_Debt_End": py_wf["Debt End"].to_numpy(dtype=float),
            "gt_FCF_for_Distribution": base["FCF_for_Distribution"].to_numpy(dtype=float),
            "py_FCF_for_Distribution": py_wf["FCF for Distribution"].to_numpy(dtype=float),
            "gt_Equity_CF": base["Equity_CF"].to_numpy(dtype=float),
            "py_Equity_CF": py_eq["Equity CF (PG)"].to_numpy(dtype=float),
        }
    )

    compare["diff_Debt_End"] = compare["py_Debt_End"] - compare["gt_Debt_End"]
    compare["diff_FCF_for_Distribution"] = (
        compare["py_FCF_for_Distribution"] - compare["gt_FCF_for_Distribution"]
    )
    compare["diff_Equity_CF"] = compare["py_Equity_CF"] - compare["gt_Equity_CF"]

    return compare


def build_metric_comparison(
    gt_df: pd.DataFrame,
    scenario: str = "Base",
    **param_overrides,
) -> pd.DataFrame:
    """
    Compare headline metrics Excel vs Python.
    """
    base = gt_df.loc[gt_df["Scenario"].astype(str).str.strip() == scenario].copy()
    params = build_params_from_ground_truth(gt_df, scenario=scenario, **param_overrides)
    series = build_series_from_ground_truth(gt_df, scenario=scenario)

    py_wf = compute_waterfall(params, series)
    py_eq = compute_pg_equity_cashflows(params, py_wf)

    gt_cf = base["Equity_CF"].to_numpy(dtype=float)
    py_cf = py_eq["Equity CF (PG)"].to_numpy(dtype=float)

    gt_total_dist = float(base["Equity_CF"].iloc[1:].sum() + max(base["Equity_CF"].iloc[0], 0.0))
    py_total_dist = float(py_cf[1:].sum() + max(py_cf[0], 0.0))

    gt_metrics = {
        "equity_ticket": float(base["Equity_Ticket"].iloc[0]),
        "pg_share": float(base["PG_Share"].iloc[0]),
        "irr": float(irr_annual(gt_cf)),
        "moic": float(moic(gt_cf)),
        "total_equity_cf": float(gt_cf.sum()),
        "total_distributions": gt_total_dist,
        "terminal_equity_cf": float(base["Equity_CF"].iloc[-1]),
        "final_debt_end": float(base["Debt_End"].iloc[-1]),
        "final_fcf_distribution": float(base["FCF_for_Distribution"].iloc[-1]),
    }

    py_metrics = {
        "equity_ticket": float(params.consortium_equity_ticket),
        "pg_share": float(params.pg_share),
        "irr": float(irr_annual(py_cf)),
        "moic": float(moic(py_cf)),
        "total_equity_cf": float(py_cf.sum()),
        "total_distributions": py_total_dist,
        "terminal_equity_cf": float(py_cf[-1]),
        "final_debt_end": float(py_wf["Debt End"].iloc[-1]),
        "final_fcf_distribution": float(py_wf["FCF for Distribution"].iloc[-1]),
    }

    rows = []
    for metric in gt_metrics.keys():
        excel_value = gt_metrics[metric]
        python_value = py_metrics[metric]
        rows.append(
            {
                "metric": metric,
                "excel_value": excel_value,
                "python_value": python_value,
                "abs_diff": python_value - excel_value,
                "rel_diff": _safe_rel_diff(python_value, excel_value),
            }
        )

    return pd.DataFrame(rows)


def build_calibration_summary(
    metric_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
    metric_tol: float = 1e-8,
    yearly_tol: float = 1e-8,
) -> pd.DataFrame:
    """
    Build a compact calibration QA summary with pass/fail checks.
    """
    max_metric_abs_diff = float(metric_df["abs_diff"].abs().max())
    max_yearly_debt_diff = float(yearly_df["diff_Debt_End"].abs().max())
    max_yearly_fcf_diff = float(yearly_df["diff_FCF_for_Distribution"].abs().max())
    max_yearly_equity_diff = float(yearly_df["diff_Equity_CF"].abs().max())

    summary_rows = [
        {
            "check": "metric_abs_diff_max",
            "value": max_metric_abs_diff,
            "tolerance": metric_tol,
            "status": "PASS" if max_metric_abs_diff <= metric_tol else "FAIL",
        },
        {
            "check": "yearly_debt_diff_max",
            "value": max_yearly_debt_diff,
            "tolerance": yearly_tol,
            "status": "PASS" if max_yearly_debt_diff <= yearly_tol else "FAIL",
        },
        {
            "check": "yearly_fcf_diff_max",
            "value": max_yearly_fcf_diff,
            "tolerance": yearly_tol,
            "status": "PASS" if max_yearly_fcf_diff <= yearly_tol else "FAIL",
        },
        {
            "check": "yearly_equity_diff_max",
            "value": max_yearly_equity_diff,
            "tolerance": yearly_tol,
            "status": "PASS" if max_yearly_equity_diff <= yearly_tol else "FAIL",
        },
    ]

    overall_pass = all(row["status"] == "PASS" for row in summary_rows)

    summary_rows.append(
        {
            "check": "overall_calibration_status",
            "value": 1.0 if overall_pass else 0.0,
            "tolerance": np.nan,
            "status": "PASS" if overall_pass else "FAIL",
        }
    )

    return pd.DataFrame(summary_rows)


def run_calibration(
    csv_path: str | Path,
    output_dir: str | Path = "outputs",
    scenario: str = "Base",
    **param_overrides,
) -> Dict[str, pd.DataFrame]:
    """
    Full calibration runner.
    """
    gt_df = load_ground_truth(csv_path)
    yearly = build_yearly_comparison(gt_df, scenario=scenario, **param_overrides)
    metrics = build_metric_comparison(gt_df, scenario=scenario, **param_overrides)
    summary = build_calibration_summary(metrics, yearly)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    yearly.to_csv(output_dir / "calibration_yearly_comparison.csv", index=False)
    metrics.to_csv(output_dir / "calibration_metric_comparison.csv", index=False)
    summary.to_csv(output_dir / "calibration_summary.csv", index=False)

    return {
        "yearly_comparison": yearly,
        "metric_comparison": metrics,
        "summary": summary,
    }