# src/pir_waterfall_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
import pandas as pd


# ---------------------------
# Dataclasses (Inputs)
# ---------------------------

@dataclass(frozen=True)
class WaterfallParams:
    """
    Parameter, die typischerweise im '01_Inputs' liegen.
    Werte in Dezimal (z.B. 5% = 0.05).
    """
    scenario: str  # "Base", "CEG", "Flat", "Downside" (oder was du nutzt)
    entry_debt: float
    recap_target_ltv: float           # z.B. 0.50
    recap_years: Tuple[int, ...]      # z.B. (2026, 2029)
    cash_sweep_start_fy: int          # z.B. 2027
    operating_fee_pct: float          # z.B. 0.05
    pg_share: float                   # z.B. 0.2954
    consortium_equity_ticket: float   # z.B. 336.5 oder 99.4 (je nach Definition)

    # Engine-Flags / Defaults
    apply_cash_sweep_only_for: Tuple[str, ...] = ("Flat", "Downside")
    terminal_override: bool = True


@dataclass(frozen=True)
class ScenarioSeries:
    """
    Zeitreihen (jährlich) – typischerweise aus '02_PIR_Data_Paste'.
    Alle Arrays müssen gleiche Länge haben.
    """
    fy: np.ndarray                    # int years
    net_cf_to_consortium: np.ndarray  # float
    interest_rate: np.ndarray         # float (dezimal, z.B. 0.062)
    nav_multiple: np.ndarray          # float
    mandatory_amort: np.ndarray       # float
    consortium_fees: np.ndarray       # float


# ---------------------------
# IRR / MOIC Helpers
# ---------------------------

def irr_newton(cashflows: np.ndarray, guess: float = 0.10) -> float:
    """
    Newton-IRR. Robust genug für typische Private/Alt-Asset CFs.
    Returns np.nan if no convergence.
    """
    r = guess
    for _ in range(200):
        npv = 0.0
        d = 0.0
        for t, cf in enumerate(cashflows):
            denom = (1.0 + r) ** t
            npv += cf / denom
            if t > 0:
                d -= t * cf / ((1.0 + r) ** (t + 1))

        if abs(npv) < 1e-10:
            return float(r)
        if d == 0:
            break

        step = npv / d
        r -= step

        if abs(step) < 1e-12:
            return float(r)
        if r <= -0.9999:
            r = -0.9999

    return float("nan")


def moic(distributions: float, equity_ticket: float) -> float:
    if equity_ticket == 0:
        return float("nan")
    return float(distributions / equity_ticket)


# ---------------------------
# Core Engine
# ---------------------------

def validate_series(s: ScenarioSeries) -> None:
    n = len(s.fy)
    fields = [
        ("net_cf_to_consortium", s.net_cf_to_consortium),
        ("interest_rate", s.interest_rate),
        ("nav_multiple", s.nav_multiple),
        ("mandatory_amort", s.mandatory_amort),
        ("consortium_fees", s.consortium_fees),
    ]
    for name, arr in fields:
        if len(arr) != n:
            raise ValueError(f"Series length mismatch: fy has {n}, {name} has {len(arr)}")


def compute_waterfall(params: WaterfallParams, series: ScenarioSeries) -> pd.DataFrame:
    """
    PIR Waterfall Engine (jährlich).

    Logik:
      OpFee = operating_fee_pct * NetCF
      CF after Fees = NetCF - OpFee - ConsortiumFees

      NAV = NetCF * NAV_multiple   (konzeptionell wie dein Excel)
      Debt Path:
        debt_begin[0] = entry_debt
        interest_cost = - debt_begin * interest_rate

        Recap in recap_years: set debt_end_pre = recap_target_ltv * NAV
        Mandatory amort reduces debt
        Cash sweep (optional) pays down debt using positive FCF pre-sweep

      FCF for Distribution = CF after Fees + interest_cost - mandatory_amort - cash_sweep

      Terminal override:
        last year FCF for Distribution = NAV_last - DebtEnd_last  (Equity Exit Value)
    """
    validate_series(series)

    fy = series.fy.astype(int)
    n = len(fy)

    op_fee = params.operating_fee_pct * series.net_cf_to_consortium
    cf_after_fees = series.net_cf_to_consortium - op_fee - series.consortium_fees
    nav = series.net_cf_to_consortium * series.nav_multiple

    debt_begin = np.zeros(n, dtype=float)
    debt_end = np.zeros(n, dtype=float)
    recap_delta_debt = np.zeros(n, dtype=float)
    interest_cost = np.zeros(n, dtype=float)
    cash_sweep = np.zeros(n, dtype=float)

    recap_set = set(params.recap_years)
    sweep_allowed = (params.scenario.strip() in set(params.apply_cash_sweep_only_for))

    for i in range(n):
        debt_begin[i] = params.entry_debt if i == 0 else debt_end[i - 1]
        interest_cost[i] = -debt_begin[i] * series.interest_rate[i]

        # Start with "no recap" end debt = begin debt
        debt_end_i = debt_begin[i]

        # Recap: set to target LTV on NAV (DebtEnd becomes LTV*NAV)
        if fy[i] in recap_set:
            target = params.recap_target_ltv * nav[i]
            recap_delta_debt[i] = target - debt_begin[i]
            debt_end_i = target

        # Mandatory amort reduces debt
        debt_end_i = max(0.0, debt_end_i - series.mandatory_amort[i])

        # Cash sweep after a start year, only in specified scenarios
        if sweep_allowed and fy[i] >= params.cash_sweep_start_fy:
            # FCF pre-sweep (after fees, after interest, after mandatory amort)
            fcf_pre_sweep = cf_after_fees[i] + interest_cost[i] - series.mandatory_amort[i]
            sweep = max(0.0, min(fcf_pre_sweep, debt_end_i))
            cash_sweep[i] = sweep
            debt_end_i = max(0.0, debt_end_i - sweep)

        debt_end[i] = debt_end_i

    fcf_for_distribution = cf_after_fees + interest_cost - series.mandatory_amort - cash_sweep

    if params.terminal_override:
        # Terminal year equity value replaces operating distribution
        fcf_for_distribution[-1] = nav[-1] - debt_end[-1]

    ltv = np.divide(debt_end, nav, out=np.zeros_like(debt_end), where=nav != 0)

    df = pd.DataFrame({
        "FY": fy,
        "Net CF to Consortium": series.net_cf_to_consortium,
        "Op Fee": op_fee,
        "Consortium Fees": series.consortium_fees,
        "CF after Fees": cf_after_fees,
        "Interest Rate": series.interest_rate,
        "Interest Cost": interest_cost,
        "Mandatory Amort": series.mandatory_amort,
        "Recap ΔDebt": recap_delta_debt,
        "Cash Sweep": cash_sweep,
        "Debt Begin": debt_begin,
        "Debt End": debt_end,
        "NAV Multiple": series.nav_multiple,
        "NAV": nav,
        "LTV": ltv,
        "FCF for Distribution": fcf_for_distribution,
    })

    return df


def compute_pg_equity_cashflows(params: WaterfallParams, waterfall: pd.DataFrame) -> pd.DataFrame:
    """
    PG-Level Equity CF:
      distributions = pg_share * FCF for Distribution
      entry = - consortium_equity_ticket at t0 (FY first)
      terminal is already included in last year's FCF (terminal_override).
    """
    dist = params.pg_share * waterfall["FCF for Distribution"].to_numpy(dtype=float)
    equity_cf = dist.copy()
    equity_cf[0] = -params.consortium_equity_ticket + dist[0]
    return pd.DataFrame({"FY": waterfall["FY"].astype(int), "Equity CF (PG)": equity_cf})


def summarize_outputs(params: WaterfallParams, waterfall: pd.DataFrame) -> Dict[str, float]:
    equity_cf_tbl = compute_pg_equity_cashflows(params, waterfall)
    equity_cf = equity_cf_tbl["Equity CF (PG)"].to_numpy(dtype=float)

    total_dist = float((params.pg_share * waterfall["FCF for Distribution"]).sum())
    irr = irr_newton(equity_cf, guess=0.10)
    multiple = moic(total_dist, params.consortium_equity_ticket)

    out = {
        "Scenario": params.scenario,
        "Equity Ticket": float(params.consortium_equity_ticket),
        "Total Distributions": float(total_dist),
        "MOIC": float(multiple),
        "IRR (annual)": float(irr),
        "Exit NAV (last FY)": float(waterfall["NAV"].iloc[-1]),
        "Exit Debt (last FY)": float(waterfall["Debt End"].iloc[-1]),
        "Max Debt": float(waterfall["Debt End"].max()),
        "Max LTV": float(waterfall["LTV"].max()),
        "Min LTV": float(waterfall["LTV"].min()),
    }
    return out