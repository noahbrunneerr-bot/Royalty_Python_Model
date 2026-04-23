import sys
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter

# =========================
# Project path setup
# =========================
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# =========================
# Imports from project
# =========================
from src.pir_waterfall_engine import (
    WaterfallParams,
    ScenarioSeries,
    compute_waterfall,
    summarize_outputs,
)
from src.metrics import monte_carlo_risk_metrics
from src.mc import run_pg3_monte_carlo as run_reference_monte_carlo, summarize_pg3_mc
from src.inputs.macro_data import get_macro_data, build_macro_scenarios

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Royalty Valuation Tool",
    layout="wide",
)

# =========================
# Color system
# =========================
BLUE = "#2563eb"
BLUE_SOFT = "#eff6ff"
GREEN = "#16a34a"
GREEN_SOFT = "#f0fdf4"
AMBER = "#d97706"
AMBER_SOFT = "#fffbeb"
RED = "#dc2626"
RED_SOFT = "#fef2f2"
SLATE = "#64748b"
TEXT = "#0f172a"
BORDER = "#e5ebf2"
LIGHT_BG = "#f8fafc"

# =========================
# Styling
# =========================
st.markdown(
    f"""
    <style>
    .block-container {{
        padding-top: 1.00rem;
        padding-bottom: 2.60rem;
        max-width: 1420px;
    }}

    div[data-testid="stMetric"] {{
        background: #ffffff;
        border: 1px solid {BORDER};
        padding: 12px 14px;
        border-radius: 12px;
    }}

    [data-testid="stMetricLabel"] {{
        font-size: 0.80rem;
        font-weight: 600;
        color: {SLATE};
    }}

    [data-testid="stMetricValue"] {{
        font-size: 1.20rem;
        font-weight: 700;
        color: {TEXT};
    }}

    .app-kicker {{
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {SLATE};
        margin-bottom: 0.18rem;
        font-weight: 700;
    }}

    .app-subtitle {{
        font-size: 14px;
        color: {SLATE};
        margin-top: 0.10rem;
        margin-bottom: 0.80rem;
    }}

    .section-gap {{
        height: 14px;
    }}

    .info-chip {{
        display:inline-block;
        padding:4px 9px;
        border-radius:999px;
        background:{LIGHT_BG};
        border:1px solid {BORDER};
        font-size:12px;
        color:#334155;
        margin-right:6px;
        margin-bottom:5px;
    }}

    .method-note {{
        background:{LIGHT_BG};
        border:1px solid {BORDER};
        border-radius:12px;
        padding:12px 14px;
        font-size:13px;
        color:#334155;
        margin-top:8px;
    }}

    .decision-summary-card {{
        border:1px solid #eadfc8;
        border-left:6px solid {AMBER};
        background:#f8f2df;
        border-radius:12px;
        padding:14px 16px;
        margin-bottom:10px;
    }}

    .decision-summary-card.invest {{
        border-left-color:{GREEN};
        background:{GREEN_SOFT};
        border-color:#d7eedb;
    }}

    .decision-summary-card.reject {{
        border-left-color:{RED};
        background:{RED_SOFT};
        border-color:#f2d0d0;
    }}

    .decision-title {{
        font-size:12px;
        text-transform:uppercase;
        letter-spacing:0.05em;
        color:#6b7280;
        margin-bottom:4px;
        font-weight:700;
    }}

    .decision-main {{
        font-size:22px;
        font-weight:800;
        color:#111827;
        line-height:1.2;
    }}

    .warning-box {{
        background:#fff7ed;
        border:1px solid #fed7aa;
        color:#9a3412;
        border-radius:12px;
        padding:12px 14px;
        margin:10px 0 16px 0;
        font-size:13px;
    }}

    .risk-flag-box {{
        background:#ffffff;
        border:1px solid {BORDER};
        border-radius:12px;
        padding:12px 14px;
        min-height:86px;
    }}

    .risk-flag-label {{
        font-size:12px;
        font-weight:700;
        color:{SLATE};
        text-transform:uppercase;
        letter-spacing:0.04em;
        margin-bottom:6px;
    }}

    .risk-flag-value {{
        font-size:13px;
        font-weight:700;
        line-height:1.30;
        word-break:break-word;
        overflow-wrap:anywhere;
    }}

    .sidebar-note {{
        font-size:12px;
        color:{SLATE};
        margin-top:0.10rem;
        margin-bottom:0.55rem;
        line-height:1.35;
    }}

    .scenario-note {{
        font-size: 12.5px;
        color: {SLATE};
        margin-top: -2px;
        margin-bottom: 14px;
    }}

    .badge-ref {{
        display:inline-block;
        padding:3px 8px;
        border-radius:999px;
        font-size:10.5px;
        font-weight:700;
        color:#475569;
        background:{LIGHT_BG};
        border:1px solid {BORDER};
        white-space:nowrap;
    }}

    .badge-invest {{
        display:inline-block;
        padding:4px 9px;
        border-radius:999px;
        font-size:10.5px;
        font-weight:700;
        color:#166534;
        background:{GREEN_SOFT};
        border:1px solid #bbf7d0;
        white-space:nowrap;
    }}

    .badge-cond {{
        display:inline-block;
        padding:4px 9px;
        border-radius:999px;
        font-size:10.5px;
        font-weight:700;
        color:#92400e;
        background:{AMBER_SOFT};
        border:1px solid #fde68a;
        white-space:nowrap;
    }}

    .badge-reject {{
        display:inline-block;
        padding:4px 9px;
        border-radius:999px;
        font-size:10.5px;
        font-weight:700;
        color:#991b1b;
        background:{RED_SOFT};
        border:1px solid #fecaca;
        white-space:nowrap;
    }}

    .scenario-card {{
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        height: auto;
        min-height: 0;
        padding: 0 0 10px 0;
    }}

    .scenario-accent {{
        height: 4px;
        border-radius: 999px;
        margin-top: -4px;
        margin-bottom: 14px;
    }}

    .scenario-content {{
        flex-grow: 0;
    }}

    .scenario-footer {{
        margin-top: 16px;
        padding-top: 8px;
    }}

    .scenario-k {{
        font-size: 10.8px;
        color: {SLATE};
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 2px;
    }}

    .scenario-v {{
        font-size: 15px;
        font-weight: 700;
        color: {TEXT};
        line-height: 1.15;
    }}

    .scenario-risk {{
        font-size: 12px;
        color: #475569;
        line-height: 1.3;
    }}

    .risk-pill {{
        display:inline-block;
        padding:3px 8px;
        border-radius:999px;
        font-size:10.5px;
        font-weight:700;
        white-space:nowrap;
        margin-top:2px;
    }}

    .risk-pill-green {{
        color:#166534;
        background:{GREEN_SOFT};
        border:1px solid #bbf7d0;
    }}

    .risk-pill-amber {{
        color:#92400e;
        background:{AMBER_SOFT};
        border:1px solid #fde68a;
    }}

    .risk-pill-red {{
        color:#991b1b;
        background:{RED_SOFT};
        border:1px solid #fecaca;
    }}

    .clean-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 12.5px;
        border: 1px solid {BORDER};
        border-radius: 12px;
        overflow: hidden;
        background: #ffffff;
        margin-top: 4px;
        margin-bottom: 10px;
    }}

    .clean-table thead th {{
        background: {LIGHT_BG};
        color: #64748b;
        font-size: 11.5px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        font-weight: 600;
        padding: 10px 12px;
        border-bottom: 1px solid {BORDER};
        text-align: left;
        white-space: nowrap;
    }}

    .clean-table tbody td {{
        padding: 10px 12px;
        border-bottom: 1px solid #eef2f7;
        white-space: nowrap;
    }}

    .clean-table tbody tr:nth-child(even) {{
        background: #fbfcfe;
    }}

    .clean-table tbody tr:last-child td {{
        border-bottom: none;
    }}

    .tbl-pill {{
        display: inline-block;
        padding: 3px 8px;
        border-radius: 999px;
        font-size: 10.5px;
        font-weight: 700;
        white-space: nowrap;
    }}

    .tbl-pill-green {{
        color: #166534;
        background: {GREEN_SOFT};
        border: 1px solid #bbf7d0;
    }}

    .tbl-pill-amber {{
        color: #92400e;
        background: {AMBER_SOFT};
        border: 1px solid #fde68a;
    }}

    .tbl-pill-red {{
        color: #991b1b;
        background: {RED_SOFT};
        border: 1px solid #fecaca;
    }}

    .driver-row {{
        display: grid;
        grid-template-columns: 1fr 230px;
        gap: 18px;
        margin-bottom: 12px;
        align-items: stretch;
    }}

    .driver-main {{
        background: #ffffff;
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 14px 16px;
    }}

    .driver-rank {{
        font-size: 11px;
        font-weight: 700;
        color: {SLATE};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }}

    .driver-title {{
        font-size: 15px;
        font-weight: 700;
        color: {TEXT};
        margin-bottom: 6px;
    }}

    .driver-desc {{
        font-size: 13px;
        color: #475569;
        line-height: 1.45;
    }}

    .driver-side {{
        background: #ffffff;
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 14px 14px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}

    .driver-side-label {{
        font-size: 11px;
        color: {SLATE};
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }}

    .driver-side-value {{
        font-size: 24px;
        font-weight: 700;
        color: {TEXT};
        line-height: 1.0;
        margin-bottom: 10px;
    }}

    .driver-bar-wrap {{
        width: 100%;
        height: 7px;
        background: #e9eef5;
        border-radius: 999px;
        overflow: hidden;
    }}

    .driver-bar-fill {{
        height: 100%;
        background: {BLUE};
        border-radius: 999px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-kicker">Investor Underwriting Interface</div>', unsafe_allow_html=True)
st.title("Royalty Valuation Tool")
st.caption(
    "Investor-grade royalty valuation with integrated Monte Carlo risk analysis and IC decision framework."
)
st.markdown(
    '<div class="app-subtitle">Standalone investor-level royalty valuation, scenario analysis and underwriting decision support.</div>',
    unsafe_allow_html=True,
)

# =========================
# Load reference dataset
# =========================
@st.cache_data
def load_reference_dataset(path: str):
    df = pd.read_csv(path)
    clean = df[df["Net_CF_to_Consortium"].notna()].copy()
    return df, clean


reference_path = project_root / "Data" / "processed" / "ground_truth_clean.csv"
reference_df, reference_clean = load_reference_dataset(str(reference_path))

fy = reference_clean["FY"].to_numpy()
net_cf = reference_clean["Net_CF_to_Consortium"].to_numpy(dtype=float)
interest_rate_arr = reference_clean["Interest_Rate"].to_numpy(dtype=float)
nav_multiple = reference_clean["NAV_Multiple"].to_numpy(dtype=float)
mandatory_amort = reference_clean["Mandatory_Amort"].to_numpy(dtype=float)
consortium_fees_arr = reference_clean["Consortium_Fees"].to_numpy(dtype=float)

reference_series = ScenarioSeries(
    fy=fy,
    net_cf_to_consortium=net_cf,
    interest_rate=interest_rate_arr,
    nav_multiple=nav_multiple,
    mandatory_amort=mandatory_amort,
    consortium_fees=consortium_fees_arr,
)

REFERENCE_INITIAL_CASHFLOW = float(reference_clean["Net_CF_to_Consortium"].iloc[0]) * 1_000_000
REFERENCE_ENTRY_MULTIPLE = float(reference_clean["NAV_Multiple"].iloc[0])
REFERENCE_ENTRY_DEBT = float(reference_clean["Debt_End"].iloc[0]) * 1_000_000
REFERENCE_INVESTOR_SHARE = float(reference_clean["PG_Share"].iloc[0])
REFERENCE_EQUITY_TICKET = float(reference_clean["Equity_Ticket"].iloc[0])
REFERENCE_ENTRY_EV = float(reference_clean["NAV_Multiple"].iloc[0]) * float(reference_clean["Net_CF_to_Consortium"].iloc[0])

# =========================
# Reference defaults only
# =========================
REFERENCE_DEFAULTS = {
    "initial_cashflow": float(REFERENCE_INITIAL_CASHFLOW),
    "growth_rate": 0.03,
    "contract_length": 15,
    "entry_multiple": float(REFERENCE_ENTRY_MULTIPLE),
    "exit_multiple": 12.0,
    "multiple_sensitivity": 1.5,
    "entry_debt": float(REFERENCE_ENTRY_DEBT),
    "ltv_target": 0.50,
    "operating_fee": 0.05,
    "consortium_fee": 0.0008,
    "valuation_discount_rate": 0.10,
    "hurdle_rate": 0.10,
    "cashflow_volatility": 0.15,
    "n_simulations": 3000,
    "investor_share": float(REFERENCE_INVESTOR_SHARE),
    "interest_rate": 0.06,
    "fund_mgmt_fee": 0.02,
    "fund_carry": 0.20,
    "use_fred": False,
    "macro_lookback_months": 12,
}

if "defaults_loaded" not in st.session_state:
    for k, v in REFERENCE_DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["defaults_loaded"] = True

# =========================
# Helpers
# =========================
def fmt_pct(x):
    if pd.isna(x):
        return "n/a"
    return f"{x:.2%}"


def fmt_x(x):
    if pd.isna(x):
        return "n/a"
    return f"{x:.2f}x"


def fmt_num(x):
    if pd.isna(x):
        return "n/a"
    return f"{x:,.2f}"


def fmt_int(x):
    if pd.isna(x):
        return "n/a"
    return f"{int(x):,}"


def safe_clip(x, low=None, high=None):
    if low is not None:
        x = max(x, low)
    if high is not None:
        x = min(x, high)
    return x


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def json_to_bytes(obj: dict) -> bytes:
    return json.dumps(obj, indent=2).encode("utf-8")


def money_formatter(x, pos):
    return f"{x:,.0f}"


def get_status_meta(decision: str):
    if decision == "INVEST":
        return {"class": "invest", "color": GREEN}
    if decision == "INVEST WITH CONDITIONS":
        return {"class": "cond", "color": AMBER}
    return {"class": "reject", "color": RED}


def get_risk_meta(risk_flag: str):
    if "ACCEPTABLE" in risk_flag:
        return {"class": "risk-pill-green", "color": GREEN}
    if "MODERATE" in risk_flag or "ELEVATED" in risk_flag:
        return {"class": "risk-pill-amber", "color": AMBER}
    return {"class": "risk-pill-red", "color": RED}


def build_clean_hist(
    data,
    title,
    xlabel,
    threshold=None,
    threshold_color=BLUE,
    mean_line=None,
    annotation=None,
):
    fig, ax = plt.subplots(figsize=(7.1, 4.0))
    ax.hist(
        data,
        bins=30,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.35,
        color=BLUE,
    )

    if threshold is not None:
        ax.axvline(threshold, linestyle="--", linewidth=1.55, color=threshold_color)

    if mean_line is not None and not pd.isna(mean_line):
        ax.axvline(mean_line, linestyle=":", linewidth=1.20, color=SLATE)

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.grid(alpha=0.08, linewidth=0.55)
    ax.tick_params(axis="both", labelsize=9)

    if annotation is not None:
        ax.text(
            0.98,
            0.92,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white", alpha=0.92, edgecolor="#d7dee7"),
        )

    fig.tight_layout()
    return fig


def format_det_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "IRR" in out.columns:
        out["IRR"] = out["IRR"].map(fmt_pct)
    if "MOIC" in out.columns:
        out["MOIC"] = out["MOIC"].map(fmt_x)
    return out


def format_mc_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pct_cols = ["IRR Mean", "IRR P10", "IRR P90", "Prob(NPV<0)"]
    x_cols = ["MOIC Mean", "MOIC P10", "MOIC P90"]
    num_cols = ["NPV Mean", "NPV P10", "NPV P90", "NPV CVaR (5%)"]

    for c in pct_cols:
        if c in out.columns:
            out[c] = out[c].map(fmt_pct)
    for c in x_cols:
        if c in out.columns:
            out[c] = out[c].map(fmt_x)
    for c in num_cols:
        if c in out.columns:
            out[c] = out[c].map(fmt_num)
    return out


def scenario_table_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Valuation Discount Rate"] = out["Valuation Discount Rate"].map(lambda x: f"{x:.2%}")
    out["Hurdle Rate"] = out["Hurdle Rate"].map(lambda x: f"{x:.2%}")
    out["Exit Multiple"] = out["Exit Multiple"].map(lambda x: f"{x:.1f}x")
    out["Volatility"] = out["Volatility"].map(lambda x: f"{x:.2%}")
    out["IRR Mean"] = out["IRR Mean"].map(lambda x: f"{x:.2%}")
    out["MOIC Mean"] = out["MOIC Mean"].map(lambda x: f"{x:.2f}x")
    out["NPV Mean"] = out["NPV Mean"].map(lambda x: f"{x:,.2f}")
    out["Prob(NPV<0)"] = out["Prob(NPV<0)"].map(lambda x: f"{x:.2%}")
    out["NPV CVaR (5%)"] = out["NPV CVaR (5%)"].map(lambda x: f"{x:,.2f}")
    return out


def discount_table_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Valuation Discount Rate"] = out["Valuation Discount Rate"].map(lambda x: f"{x:.2%}")
    out["IRR Mean"] = out["IRR Mean"].map(lambda x: f"{x:.2%}")
    out["MOIC Mean"] = out["MOIC Mean"].map(lambda x: f"{x:.2f}x")
    out["NPV Mean"] = out["NPV Mean"].map(lambda x: f"{x:,.2f}")
    out["NPV CVaR (5%)"] = out["NPV CVaR (5%)"].map(lambda x: f"{x:,.2f}")
    out["Prob(NPV<0)"] = out["Prob(NPV<0)"].map(lambda x: f"{x:.2%}")
    return out


def render_clean_table(df: pd.DataFrame, decision_col: str = None, risk_col: str = None):
    html_df = df.copy()

    if decision_col and decision_col in html_df.columns:
        def style_decision(x):
            if x == "INVEST":
                return f'<span class="tbl-pill tbl-pill-green">{x}</span>'
            if x == "INVEST WITH CONDITIONS":
                return f'<span class="tbl-pill tbl-pill-amber">{x}</span>'
            return f'<span class="tbl-pill tbl-pill-red">{x}</span>'
        html_df[decision_col] = html_df[decision_col].map(style_decision)

    if risk_col and risk_col in html_df.columns:
        def style_risk(x):
            if "ACCEPTABLE" in x:
                return f'<span class="tbl-pill tbl-pill-green">{x}</span>'
            if "MODERATE" in x or "ELEVATED" in x:
                return f'<span class="tbl-pill tbl-pill-amber">{x}</span>'
            return f'<span class="tbl-pill tbl-pill-red">{x}</span>'
        html_df[risk_col] = html_df[risk_col].map(style_risk)

    table_html = html_df.to_html(index=False, escape=False, classes="clean-table")
    st.markdown(table_html, unsafe_allow_html=True)


def build_plausibility_warnings(
    entry_multiple,
    valuation_discount_rate,
    hurdle_rate,
    ltv_target,
    interest_rate,
    cashflow_volatility,
):
    warnings = []

    if ltv_target > 0.60:
        warnings.append("LTV target exceeds 60%, which is aggressive for a royalty-style underwriting case.")
    if entry_multiple > 16.0:
        warnings.append("Entry multiple is above 16.0x and may imply stretched pricing relative to conservative underwriting.")
    if valuation_discount_rate < interest_rate + 0.02:
        warnings.append("Valuation discount rate is less than interest rate + 2%, which may understate required return / risk premium.")
    if hurdle_rate < valuation_discount_rate:
        warnings.append("Hurdle rate is below valuation discount rate. This weakens decision discipline.")
    if cashflow_volatility > 0.30:
        warnings.append("Cashflow volatility is very high and may indicate a concentrated or structurally unstable case.")

    return warnings


def render_risk_flag_box(risk_flag: str):
    risk_meta = get_risk_meta(risk_flag)
    st.markdown(
        f"""
        <div class="risk-flag-box">
            <div class="risk-flag-label">Risk Flag</div>
            <div class="risk-flag-value" style="color:{risk_meta['color']};">{risk_flag}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(decision: str):
    meta = get_status_meta(decision)
    if meta["class"] == "invest":
        st.markdown('<span class="badge-invest">INVEST</span>', unsafe_allow_html=True)
    elif meta["class"] == "cond":
        st.markdown('<span class="badge-cond">INVEST WITH CONDITIONS</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-reject">REJECT</span>', unsafe_allow_html=True)


def render_scenario_card(row: pd.Series, base_irr: float):
    status_meta = get_status_meta(row["Decision"])
    risk_meta = get_risk_meta(row["Risk Flag"])

    with st.container(border=True):
        st.markdown('<div class="scenario-card">', unsafe_allow_html=True)

        st.markdown(
            f'<div class="scenario-accent" style="background:{status_meta["color"]};"></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="scenario-content">', unsafe_allow_html=True)

        top_l, top_r = st.columns([4.2, 1.0])
        with top_l:
            st.markdown(f"**{row['Scenario']}**")
        with top_r:
            if row["Scenario"] == "Base":
                st.markdown('<span class="badge-ref">Ref.</span>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style="
                min-height: 38px;
                font-size: 12.5px;
                color: #6b7280;
                line-height: 1.45;
                margin-bottom: 12px;
                overflow: hidden;
            ">
                {row["Scenario Label"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_status_badge(row["Decision"])
        st.markdown("")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="scenario-k">IRR Mean</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="scenario-v">{fmt_pct(row["IRR Mean"])}</div>', unsafe_allow_html=True)
        
            delta_irr = row["IRR Mean"] - base_irr
            st.markdown(
                f'<div style="font-size:11px; color:#6b7280; margin-top:3px;">Δ vs Base: {delta_irr:+.2%}</div>',
                unsafe_allow_html=True,
            )
        
        with c2:
            st.markdown('<div class="scenario-k">MOIC Mean</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="scenario-v">{fmt_x(row["MOIC Mean"])}</div>', unsafe_allow_html=True)wn(f'<div class="scenario-v">{fmt_x(row["MOIC Mean"])}</div>', unsafe_allow_html=True)

        st.markdown("")

        c3, c4 = st.columns(2)
        with c3:
            st.markdown('<div class="scenario-k">NPV Mean</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="scenario-v">{fmt_num(row["NPV Mean"])}</div>', unsafe_allow_html=True)
        
            delta_npv = row["NPV_Delta_vs_Base"]
            st.markdown(
                f'<div style="font-size:11px; color:#6b7280; margin-top:3px;">Δ vs Base: {delta_npv:+.2f}</div>',
                unsafe_allow_html=True,
            )
        
        with c4:
            st.markdown('<div class="scenario-k">Prob(NPV<0)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="scenario-v">{fmt_pct(row["Prob(NPV<0)"])}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="scenario-footer">
                <div class="scenario-risk">
                    <b>Risk:</b>
                    <span class="risk-pill {risk_meta['class']}">{row["Risk Flag"]}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Sidebar
# =========================
st.sidebar.header("Model Inputs")
st.sidebar.markdown(
    '<div class="sidebar-note">Reference-case defaults are loaded automatically and can be overridden manually.</div>',
    unsafe_allow_html=True,
)

with st.sidebar.expander("Core Assumptions", expanded=True):
    initial_cashflow = st.number_input(
        "Initial Cashflow",
        min_value=0.0,
        value=float(st.session_state["initial_cashflow"]),
        step=1_000_000.0,
    )
    growth_rate = st.number_input(
        "Growth Rate",
        value=float(st.session_state["growth_rate"]),
        step=0.005,
        format="%.3f",
    )
    contract_length = st.number_input(
        "Contract Length",
        min_value=1,
        value=int(st.session_state["contract_length"]),
        step=1,
    )
    entry_multiple = st.number_input(
        "Entry Multiple",
        min_value=0.1,
        value=float(st.session_state["entry_multiple"]),
        step=0.1,
    )
    exit_multiple = st.number_input(
        "Exit Multiple",
        min_value=0.1,
        value=float(st.session_state["exit_multiple"]),
        step=0.1,
    )
    entry_debt = st.number_input(
        "Entry Debt",
        min_value=0.0,
        value=float(st.session_state["entry_debt"]),
        step=5_000_000.0,
    )
    ltv_target = st.number_input(
        "LTV Target",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state["ltv_target"]),
        step=0.01,
        format="%.2f",
    )
    operating_fee = st.number_input(
        "Operating Fee",
        min_value=0.0,
        value=float(st.session_state["operating_fee"]),
        step=0.005,
        format="%.3f",
    )
    consortium_fee = st.number_input(
        "Consortium Fee",
        min_value=0.0,
        value=float(st.session_state["consortium_fee"]),
        step=0.0001,
        format="%.4f",
    )
    valuation_discount_rate = st.number_input(
        "Valuation Discount Rate",
        min_value=0.0,
        value=float(st.session_state["valuation_discount_rate"]),
        step=0.005,
        format="%.3f",
    )
    hurdle_rate = st.number_input(
        "Hurdle Rate",
        min_value=0.0,
        value=float(st.session_state["hurdle_rate"]),
        step=0.005,
        format="%.3f",
    )
    cashflow_volatility = st.number_input(
        "Cashflow Volatility",
        min_value=0.0,
        value=float(st.session_state["cashflow_volatility"]),
        step=0.01,
        format="%.2f",
    )
    n_simulations = st.number_input(
        "Monte Carlo Runs",
        min_value=100,
        value=int(st.session_state["n_simulations"]),
        step=500,
    )
    investor_share = st.number_input(
        "Investor Share",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state["investor_share"]),
        step=0.01,
        format="%.4f",
    )
    interest_rate = st.number_input(
        "Interest Rate",
        min_value=0.0,
        value=float(st.session_state["interest_rate"]),
        step=0.005,
        format="%.3f",
    )

with st.sidebar.expander("Scenario Layer", expanded=False):
    upside_rate_delta = st.number_input("Upside: Valuation Rate Δ", value=-0.01, step=0.005, format="%.3f")
    upside_exit_delta = st.number_input("Upside: Exit Multiple Δ", value=0.5, step=0.1)
    upside_vol_delta = st.number_input("Upside: Volatility Δ", value=-0.02, step=0.01, format="%.2f")

    high_rate_delta = st.number_input("High Rate: Valuation Rate Δ", value=0.02, step=0.005, format="%.3f")
    high_rate_exit_delta = st.number_input("High Rate: Exit Multiple Δ", value=-1.0, step=0.1)
    high_rate_vol_delta = st.number_input("High Rate: Volatility Δ", value=0.03, step=0.01, format="%.2f")

    stress_rate_delta = st.number_input("Stress: Valuation Rate Δ", value=0.03, step=0.005, format="%.3f")
    stress_exit_delta = st.number_input("Stress: Exit Multiple Δ", value=-1.5, step=0.1)
    stress_vol_delta = st.number_input("Stress: Volatility Δ", value=0.05, step=0.01, format="%.2f")
    multiple_sensitivity = st.number_input(
        "Multiple Sensitivity",
        min_value=0.5,
        max_value=3.0,
        value=float(st.session_state.get("multiple_sensitivity", 1.5)),
        step=0.1,
        format="%.1f",
        help="Controls how strongly exit multiples compress or expand as macro-implied valuation rates move.",
    )

with st.sidebar.expander("Optional Stylised Downside Overlay", expanded=False):
    use_tail_overlay = st.checkbox("Use stylised tail / super-tail overlay", value=False)
    tail_event_prob = st.number_input("Tail Event Probability", value=0.03, step=0.01, format="%.2f")
    tail_event_severity = st.number_input("Tail Event Severity", value=0.20, step=0.05, format="%.2f")
    super_tail_prob = st.number_input("Super-Tail Probability", value=0.01, step=0.005, format="%.3f")
    super_tail_severity = st.number_input("Super-Tail Severity", value=0.40, step=0.05, format="%.2f")

    st.caption(
        "Prototype overlay only. Primary downside should be judged through volatility, scenario stress and valuation sensitivity."
    )

with st.sidebar.expander("Illustrative Fund-Level Bridge", expanded=False):
    fund_mgmt_fee = st.number_input(
        "Mgmt Fee Drag",
        min_value=0.0,
        value=float(st.session_state["fund_mgmt_fee"]),
        step=0.005,
        format="%.3f",
    )
    fund_carry = st.number_input(
        "Carry Rate",
        min_value=0.0,
        max_value=0.99,
        value=float(st.session_state["fund_carry"]),
        step=0.05,
        format="%.2f",
    )

with st.sidebar.expander("Macro / FRED", expanded=False):
    use_fred = st.checkbox(
        "Use FRED live data",
        value=bool(st.session_state.get("use_fred", False)),
    )
    macro_lookback_months = st.number_input(
        "Macro Lookback Months",
        min_value=3,
        max_value=60,
        value=int(st.session_state.get("macro_lookback_months", 12)),
        step=1,
    )

    st.caption(
        "If enabled, the app tries to pull live macro data from FRED. If unavailable, it falls back to the local CSV sample."
    )

run_button = st.sidebar.button("Run Valuation", type="primary")

# =========================
# Simulation engine
# =========================
def make_secondary_config(base_config):
    cfg = base_config.copy()
    cfg["n_simulations"] = max(700, min(1600, int(base_config["n_simulations"] / 4)))
    return cfg

def build_macro_base_config(base_config):

    use_fred = base_config.get("use_fred", False)

    macro_df, macro_source = get_macro_data(use_fred=use_fred)

    macro_hist_scenarios = build_macro_scenarios(
        macro_df,
        use_recent_n=base_config.get("macro_lookback_months", 12),
    )

    base_macro_row = macro_hist_scenarios.loc[
        macro_hist_scenarios["Scenario"] == "Base"
    ].iloc[0]

    macro_base_rate = float(base_macro_row["discount_rate"])

    cfg = base_config.copy()
    cfg["discount_rate"] = macro_base_rate
    cfg["valuation_discount_rate"] = macro_base_rate
    cfg["macro_source"] = macro_source

    fed_funds = macro_df["fed_funds"].iloc[-1] if "fed_funds" in macro_df.columns else None
    cfg["fed_funds"] = fed_funds

    return cfg, macro_df, macro_hist_scenarios, macro_source, macro_base_rate


def apply_tail_super_tail_overlay(
    mc_out,
    tail_prob,
    tail_severity,
    super_tail_prob,
    super_tail_severity,
    random_state=123,
):
    rng = np.random.default_rng(random_state)
    n = len(mc_out["npv"])
    draws = rng.random(n)

    super_mask = draws < super_tail_prob
    tail_mask = (draws >= super_tail_prob) & (draws < super_tail_prob + tail_prob)

    if tail_mask.any():
        mc_out["npv"][tail_mask] = mc_out["npv"][tail_mask] - np.maximum(np.abs(mc_out["npv"][tail_mask]), 1.0) * tail_severity
        mc_out["moic"][tail_mask] = np.maximum(0.0, mc_out["moic"][tail_mask] * (1 - 0.60 * tail_severity))
        mc_out["irr"][tail_mask] = mc_out["irr"][tail_mask] - 0.20 * tail_severity
        if "net_cf" in mc_out:
            try:
                mc_out["net_cf"][tail_mask] = mc_out["net_cf"][tail_mask] * (1 - tail_severity)
            except Exception:
                pass

    if super_mask.any():
        mc_out["npv"][super_mask] = mc_out["npv"][super_mask] - np.maximum(np.abs(mc_out["npv"][super_mask]), 1.0) * super_tail_severity
        mc_out["moic"][super_mask] = np.maximum(0.0, mc_out["moic"][super_mask] * (1 - 0.75 * super_tail_severity))
        mc_out["irr"][super_mask] = mc_out["irr"][super_mask] - 0.25 * super_tail_severity
        if "net_cf" in mc_out:
            try:
                mc_out["net_cf"][super_mask] = mc_out["net_cf"][super_mask] * (1 - super_tail_severity)
            except Exception:
                pass

    overlay_stats = {
        "tail_event_count": int(tail_mask.sum()),
        "super_tail_event_count": int(super_mask.sum()),
    }
    return mc_out, overlay_stats


def run_sim(config, base_df):
    nsim = int(config["n_simulations"])
    sigma_cf = float(config["sigma_cf"])
    sigma_multiple = float(config.get("sigma_multiple", 0.20))

    valuation_discount_rate_used = float(config["valuation_discount_rate"])
    hurdle_rate_used = float(config["hurdle_rate"])

    exit_multiple_used = float(config["exit_multiple"])
    operating_fee_rate = float(config["operating_fee"])
    consortium_fee_rate = float(config["consortium_fee"])
    interest_rate_used = float(config["interest_rate"])

    entry_ev = float(config["entry_ev"])
    entry_debt_used = float(config["entry_debt"])
    investor_share_used = float(config["investor_share"])
    ltv_target_used = float(config["ltv_target"])

    tail_event_prob_used = float(config.get("tail_event_prob", 0.0))
    tail_event_severity_used = float(config.get("tail_event_severity", 0.0))
    super_tail_prob_used = float(config.get("super_tail_prob", 0.0))
    super_tail_severity_used = float(config.get("super_tail_severity", 0.0))

    fund_mgmt_fee_used = float(config.get("fund_mgmt_fee", 0.0))
    fund_carry_used = float(config.get("fund_carry", 0.0))
    illustrative_gross_target_used = config.get("illustrative_gross_target_return", np.nan)

    unit_scale = 1_000_000.0

    entry_ev_tmp = entry_ev / unit_scale
    entry_debt_tmp = entry_debt_used / unit_scale
    entry_equity_total_tmp = entry_ev_tmp - entry_debt_tmp
    entry_equity_investor_tmp = entry_equity_total_tmp * investor_share_used

    params_tmp = WaterfallParams(
        scenario=config.get("scenario_name", "Reference_Run"),
        entry_debt=entry_debt_tmp,
        recap_target_ltv=ltv_target_used,
        recap_years=(fy[3], fy[6]),
        cash_sweep_start_fy=fy[2],
        operating_fee_pct=operating_fee_rate,
        pg_share=investor_share_used,
        consortium_equity_ticket=entry_equity_investor_tmp,
    )

    wf_tmp = compute_waterfall(params_tmp, reference_series)
    outs_tmp = summarize_outputs(params_tmp, wf_tmp)

    det_summary = {
        "IRR": outs_tmp.get("IRR (annual)", np.nan),
        "MOIC": outs_tmp.get("MOIC", np.nan),
    }

    mc_out = run_reference_monte_carlo(
        base_df=base_df,
        exit_multiple_mean=exit_multiple_used,
        exit_multiple_sigma=sigma_multiple,
        n_sim=nsim,
        random_state=42,
        discount_rate=valuation_discount_rate_used,
        sigma_cf=sigma_cf,
        operating_fee_rate=operating_fee_rate,
        consortium_fee_rate=consortium_fee_rate,
        interest_rate=interest_rate_used,
        amort_rate=0.0,
        cash_sweep_rate=0.0,
        recap_year=None,
        recap_amount=0.0,
    )

    if config.get("use_tail_overlay", False):
        mc_out, overlay_stats = apply_tail_super_tail_overlay(
            mc_out=mc_out,
            tail_prob=tail_event_prob_used,
            tail_severity=tail_event_severity_used,
            super_tail_prob=super_tail_prob_used,
            super_tail_severity=super_tail_severity_used,
            random_state=123,
        )
    else:
        overlay_stats = {
            "tail_event_count": 0,
            "super_tail_event_count": 0,
        }

    mc_summary = summarize_pg3_mc(mc_out)

    risk = monte_carlo_risk_metrics(
        {
            "irr": mc_out["irr"],
            "moic": mc_out["moic"],
            "net_cf": mc_out["net_cf"],
            "npv": mc_out["npv"],
            "payback_period": np.full_like(mc_out["irr"], np.nan, dtype=float),
        },
        alpha=0.05,
    )

    return {
        "deterministic": det_summary,
        "monte_carlo": mc_summary,
        "risk": risk,
        "tail_overlay": overlay_stats,
        "raw_mc": mc_out,
        "used_config": {
            "preset": "Reference Defaults / Manual Override",
            "run_timestamp_utc": datetime.utcnow().isoformat(),
            "n_simulations": nsim,
            "sigma_cf": sigma_cf,
            "sigma_multiple": sigma_multiple,
            "valuation_discount_rate": valuation_discount_rate_used,
            "hurdle_rate": hurdle_rate_used,
            "exit_multiple": exit_multiple_used,
            "operating_fee": operating_fee_rate,
            "consortium_fee": consortium_fee_rate,
            "interest_rate": interest_rate_used,
            "entry_ev": entry_ev,
            "entry_debt": entry_debt_used,
            "ltv_target": ltv_target_used,
            "investor_share": investor_share_used,
            "tail_event_prob": tail_event_prob_used,
            "tail_event_severity": tail_event_severity_used,
            "super_tail_prob": super_tail_prob_used,
            "super_tail_severity": super_tail_severity_used,
            "fund_mgmt_fee": fund_mgmt_fee_used,
            "fund_carry": fund_carry_used,
            "illustrative_gross_target_return": illustrative_gross_target_used,
                        "use_fred": config.get("use_fred", False),
                        "macro_lookback_months": config.get("macro_lookback_months", None),
                        "macro_source": config.get("macro_source", None),
        },
    }

# =========================
# Decision logic
# =========================
def make_decision(mc, risk, hurdle_rate_used):
    irr_mean = mc.get("irr_mean", np.nan)
    moic_mean = mc.get("moic_mean", np.nan)
    npv_mean = mc.get("npv_mean", np.nan)
    irr_p10 = mc.get("irr_p10", np.nan)
    prob_neg = risk.get("prob_npv_negative", np.nan)
    npv_cvar = risk.get("npv_cvar_5", np.nan)

    hurdle_spread = irr_mean - hurdle_rate_used

    gate_irr_ok = irr_mean >= hurdle_rate_used
    gate_moic_ok = moic_mean >= 2.0
    gate_prob_ok = prob_neg < 0.40
    gate_cvar_ok = npv_cvar > -20
    hard_gate_pass = all([gate_irr_ok, gate_moic_ok, gate_prob_ok, gate_cvar_ok])

    return_score = 0
    if irr_mean >= hurdle_rate_used + 0.03:
        return_score += 3
    elif irr_mean >= hurdle_rate_used + 0.01:
        return_score += 2
    elif irr_mean >= hurdle_rate_used:
        return_score += 1

    if moic_mean >= 3.0:
        return_score += 3
    elif moic_mean >= 2.5:
        return_score += 2
    elif moic_mean >= 2.0:
        return_score += 1

    if npv_mean > 20:
        return_score += 2
    elif npv_mean > 0:
        return_score += 1

    risk_score = 0
    if prob_neg <= 0.10:
        risk_score += 3
    elif prob_neg <= 0.20:
        risk_score += 2
    elif prob_neg <= 0.35:
        risk_score += 1

    if npv_cvar > -10:
        risk_score += 3
    elif npv_cvar > -20:
        risk_score += 2
    elif npv_cvar > -30:
        risk_score += 1

    resilience_score = 0
    if not pd.isna(irr_p10):
        if irr_p10 >= hurdle_rate_used - 0.01:
            resilience_score += 2
        elif irr_p10 >= hurdle_rate_used - 0.03:
            resilience_score += 1

    if hurdle_spread >= 0.02:
        resilience_score += 2
    elif hurdle_spread >= 0.00:
        resilience_score += 1

    total_score = return_score + risk_score + resilience_score

    if hard_gate_pass and total_score >= 9 and prob_neg <= 0.25 and npv_cvar > -15:
        final_decision = "INVEST"
    elif total_score >= 5 and prob_neg <= 0.60:
        final_decision = "INVEST WITH CONDITIONS"
    else:
        final_decision = "REJECT"

    if prob_neg > 0.60 or npv_cvar <= -30:
        risk_flag = "HIGH DOWNSIDE RISK"
    elif prob_neg > 0.35 or npv_cvar <= -20:
        risk_flag = "ELEVATED DOWNSIDE RISK"
    elif prob_neg > 0.20 or npv_cvar <= -10:
        risk_flag = "MODERATE DOWNSIDE RISK"
    else:
        risk_flag = "ACCEPTABLE DOWNSIDE RISK"

    # Interpretation
    if final_decision == "INVEST":
        if return_score >= 4 and risk_score >= 4:
            interpretation = (
                f"Attractive risk-return profile: IRR exceeds the hurdle by {hurdle_spread:.2%} "
                f"with contained downside risk (Prob<0 = {prob_neg:.1%}, CVaR = {npv_cvar:.2f})."
            )
        else:
            interpretation = (
                f"Investment case clears minimum thresholds, but excess return remains limited "
                f"(spread {hurdle_spread:.2%}), requiring disciplined entry valuation."
            )

    elif final_decision == "INVEST WITH CONDITIONS":
        interpretation = (
            "The case is conditionally investable, but valuation sensitivity, downside probability or tail-risk metrics require tighter underwriting discipline and explicit deal protections."
        )

    else:
        interpretation = (
            "The case does not currently satisfy the required return / downside balance under the applied underwriting assumptions and should not proceed without material repricing or structural improvement."
        )

    return {
        "FINAL_DECISION": final_decision,
        "Risk_Flag": risk_flag,
        "Interpretation": interpretation,
        "Hard_Gates_Passed": hard_gate_pass,
        "Return_Score": return_score,
        "Risk_Score": risk_score,
        "Resilience_Score": resilience_score,
        "Total_Score": total_score,
        "Gate_IRR": gate_irr_ok,
        "Gate_MOIC": gate_moic_ok,
        "Gate_ProbNeg": gate_prob_ok,
        "Gate_CVaR": gate_cvar_ok,
        "Hurdle_Spread": hurdle_spread,
    }


def build_underwriting_reasons(mc, risk, hurdle_rate_used, primary_driver):
    irr = mc["irr_mean"]
    moic = mc["moic_mean"]
    npv_mean = mc["npv_mean"]
    prob_neg = risk["prob_npv_negative"]
    npv_cvar = risk["npv_cvar_5"]

    reasons = []

    hurdle_spread = irr - hurdle_rate_used
    if hurdle_spread >= 0.02:
        reasons.append(f"Return exceeds the hurdle with a strong spread of {hurdle_spread:+.2%}.")
    elif hurdle_spread >= 0:
        reasons.append(f"Return clears the hurdle by {hurdle_spread:+.2%}, but with limited excess buffer.")
    else:
        reasons.append(f"Return misses the hurdle by {hurdle_spread:+.2%}, which weakens standalone investability.")

    if moic >= 2.5:
        reasons.append(f"Value creation remains supportive with a mean MOIC of {moic:.2f}x.")
    elif moic >= 2.0:
        reasons.append(f"Mean MOIC of {moic:.2f}x is acceptable, but not materially above minimum return expectations.")
    else:
        reasons.append(f"Mean MOIC of {moic:.2f}x remains below a robust underwriting comfort zone.")

    if npv_mean > 20:
        reasons.append(f"Expected NPV of {npv_mean:.2f} provides a meaningful valuation cushion.")
    elif npv_mean > 0:
        reasons.append(f"Expected NPV remains positive at {npv_mean:.2f}, but the valuation cushion is modest.")
    else:
        reasons.append(f"Expected NPV is negative at {npv_mean:.2f}, indicating insufficient value support at current assumptions.")

    if prob_neg <= 0.20:
        reasons.append(f"Downside probability remains contained at {prob_neg:.2%}.")
    elif prob_neg <= 0.40:
        reasons.append(f"Downside probability of {prob_neg:.2%} is elevated and should be monitored closely.")
    else:
        reasons.append(f"Downside probability is high at {prob_neg:.2%}, which materially weakens robustness.")

    if npv_cvar > -10:
        reasons.append(f"Tail-risk remains manageable with NPV CVaR (5%) of {npv_cvar:.2f}.")
    elif npv_cvar > -20:
        reasons.append(f"Tail-risk is meaningful with NPV CVaR (5%) of {npv_cvar:.2f}, but still within a conditional underwriting range.")
    else:
        reasons.append(f"Tail-risk is severe with NPV CVaR (5%) of {npv_cvar:.2f}.")

    reasons.append(f"Primary value driver is currently {primary_driver}.")

    if primary_driver == "Scenario Layer":
        reasons.append("Overall robustness is driven primarily by coordinated shifts in discount rate, exit multiple and volatility.")
    elif primary_driver == "Valuation Discount Rate":
        reasons.append("The case is especially exposed to required-return assumptions and long-duration valuation effects.")
    else:
        reasons.append("Structural economics, payout timing and contract mechanics remain material to value creation and downside protection.")

    return reasons

def build_why_invest_lists(mc, risk, decision, primary_driver, hurdle_rate_used):
    positive_points = []
    caution_points = []

    irr_mean = mc["irr_mean"]
    moic_mean = mc["moic_mean"]
    npv_mean = mc["npv_mean"]
    prob_neg = risk["prob_npv_negative"]
    npv_cvar = risk["npv_cvar_5"]
    hurdle_spread = irr_mean - hurdle_rate_used

    
    # Positive points
    if hurdle_spread > 0:
        positive_points.append(f"Return exceeds hurdle by {hurdle_spread:.2%} (tight but positive).")

    if moic_mean >= 2.5:
        positive_points.append(f"Strong capital efficiency with MOIC {moic_mean:.2f}x.")
    elif moic_mean >= 2.0:
        positive_points.append(f"Acceptable capital efficiency with MOIC {moic_mean:.2f}x.")

    if npv_mean > 0:
        positive_points.append(f"Positive valuation cushion (NPV {npv_mean:.2f}).")

    if prob_neg <= 0.05:
        positive_points.append(f"Low downside probability ({prob_neg:.2%}).")
    elif prob_neg <= 0.20:
        positive_points.append(f"Moderate downside probability ({prob_neg:.2%}).")

    if npv_cvar > -10:
        positive_points.append(f"Contained tail risk (CVaR {npv_cvar:.2f}).")
    elif npv_cvar > -15:
        positive_points.append(f"Manageable tail risk (CVaR {npv_cvar:.2f}).")

    # Caution points
    if hurdle_spread <= 0.01:
        caution_points.append("Limited excess return buffer vs hurdle.")

    if primary_driver == "Valuation Discount Rate":
        caution_points.append("Required return assumptions materially affect valuation.")
    elif primary_driver == "Scenario Layer":
        caution_points.append("Coordinated macro shifts can materially weaken robustness.")
    else:
        caution_points.append("Contract structure remains a relevant downside driver.")

    if prob_neg >= 0.20:
        caution_points.append(f"Elevated downside probability ({prob_neg:.2%}).")

    if npv_cvar <= -15:
        caution_points.append(f"Meaningful tail risk (CVaR {npv_cvar:.2f}).")

    if decision["FINAL_DECISION"] == "INVEST WITH CONDITIONS":
        caution_points.append("Case remains investable, but only with conditions.")
    elif decision["FINAL_DECISION"] == "REJECT":
        caution_points.append("Current assumptions do not support standalone investability.")

    return positive_points, caution_points
    

def build_ic_summary_text(decision, mc, risk, macro_base_rate, hurdle_rate_used, macro_source, fed_display, underwriting_reasons):
    hard_gates = "Yes" if decision.get("Hard_Gates_Passed", False) else "No"

    reason_block = "\n".join([f"- {r}" for r in underwriting_reasons])

    text = f"""
INVESTMENT COMMITTEE SUMMARY

Recommendation
{decision["FINAL_DECISION"]}

Risk Assessment
{decision["Risk_Flag"]}

Decision Context
Hard gates passed: {hard_gates}
Return score: {decision["Return_Score"]}
Risk score: {decision["Risk_Score"]}
Resilience score: {decision.get("Resilience_Score", "n/a")}
Total score: {decision["Total_Score"]}

Key Metrics
IRR mean: {mc["irr_mean"]:.2%}
IRR P10: {mc.get("irr_p10", np.nan):.2%}
MOIC mean: {mc["moic_mean"]:.2f}x
NPV mean: {mc["npv_mean"]:.2f}
Probability of negative NPV: {risk["prob_npv_negative"]:.2%}
NPV CVaR (5%): {risk["npv_cvar_5"]:.2f}
Hurdle spread: {decision.get("Hurdle_Spread", np.nan):+.2%}

Macro / Valuation Framing
Macro source: {macro_source}
Fed Funds: {fed_display}
Macro-implied base valuation rate: {macro_base_rate:.2%}
Hurdle rate: {hurdle_rate_used:.2%}

Interpretation
{decision["Interpretation"]}

Underwriting Rationale
{reason_block}
""".strip()

    return text

# =========================
# Secondary analyses
# =========================
def build_scenario_table(base_config, base_df, base_mc, base_risk, base_decision):
    macro_cfg, macro_df, macro_hist_scenarios, macro_source, macro_base_rate = build_macro_base_config(base_config)

    base_exit_multiple = float(base_config["exit_multiple"])

    # Neutral rate (deine Annahme)
    neutral_rate = 0.08  # 8% typische required return

    # Wie sensitiv Multiples reagieren
    sensitivity = float(base_config.get("multiple_sensitivity", 1.5))

    rate_diff = macro_base_rate - neutral_rate

    exit_shift = 1 - sensitivity * rate_diff

    macro_exit_multiple = safe_clip(base_exit_multiple * exit_shift, low=1.0)


    scenario_inputs = [
        {
            "Scenario": "Upside",
            "Scenario Label": "Lower-rate, higher-multiple, lower-volatility case.",
            "Valuation Discount Rate": max(0.01, macro_base_rate - 0.01),
            "Exit Multiple": safe_clip(macro_exit_multiple * 1.05, low=1.0),
            "Volatility": safe_clip(base_config["sigma_cf"] - 0.02, low=0.01),
        },
        {
            "Scenario": "Base",
            "Scenario Label": "Reference underwriting assumptions.",
            "Valuation Discount Rate": macro_base_rate,
            "Exit Multiple": macro_exit_multiple,
            "Volatility": base_config["sigma_cf"],
        },
        {
            "Scenario": "High-Rate",
            "Scenario Label": "Higher discount rate environment (macro tightening).",
            "Valuation Discount Rate": macro_base_rate + 0.015,
            "Exit Multiple": macro_exit_multiple,
            "Volatility": base_config["sigma_cf"],
        },
        {
            "Scenario": "Stress",
            "Scenario Label": "Severe downside case with multiple compression and volatility shock.",
            "Valuation Discount Rate": macro_base_rate + 0.04,
            "Exit Multiple": safe_clip(macro_exit_multiple * 0.85, low=1.0),
            "Volatility": safe_clip(base_config["sigma_cf"] + 0.04, low=0.01),
        },
    ]

    rows = []
    quick_cfg = make_secondary_config(macro_cfg)

    for s in scenario_inputs:

        if s["Scenario"] == "Base":
            rows.append({
                "Scenario": s["Scenario"],
                "Scenario Label": s["Scenario Label"],
                "Valuation Discount Rate": macro_base_rate,
                "Hurdle Rate": base_config["hurdle_rate"],
                "Exit Multiple": macro_exit_multiple,
                "Volatility": base_config["sigma_cf"],
                "IRR Mean": base_mc["irr_mean"],
                "MOIC Mean": base_mc["moic_mean"],
                "NPV Mean": base_mc["npv_mean"],
                "Prob(NPV<0)": base_risk["prob_npv_negative"],
                "NPV CVaR (5%)": base_risk["npv_cvar_5"],
                "Decision": base_decision["FINAL_DECISION"],
                "Risk Flag": base_decision["Risk_Flag"],
            })
            continue

        cfg = quick_cfg.copy()
        cfg["discount_rate"] = s["Valuation Discount Rate"]
        cfg["valuation_discount_rate"] = s["Valuation Discount Rate"]
        cfg["hurdle_rate"] = base_config["hurdle_rate"]
        cfg["exit_multiple"] = s["Exit Multiple"]
        cfg["sigma_cf"] = s["Volatility"]
        cfg["scenario_name"] = s["Scenario"]

        res = run_sim(cfg, base_df)
        dec = make_decision(res["monte_carlo"], res["risk"], cfg["hurdle_rate"])

        rows.append({
            "Scenario": s["Scenario"],
            "Scenario Label": s["Scenario Label"],
            "Valuation Discount Rate": cfg["valuation_discount_rate"],
            "Hurdle Rate": cfg["hurdle_rate"],
            "Exit Multiple": cfg["exit_multiple"],
            "Volatility": cfg["sigma_cf"],
            "IRR Mean": res["monte_carlo"]["irr_mean"],
            "MOIC Mean": res["monte_carlo"]["moic_mean"],
            "NPV Mean": res["monte_carlo"]["npv_mean"],
            "Prob(NPV<0)": res["risk"]["prob_npv_negative"],
            "NPV CVaR (5%)": res["risk"]["npv_cvar_5"],
            "Decision": dec["FINAL_DECISION"],
            "Risk Flag": dec["Risk_Flag"],
        })

    scenario_df = pd.DataFrame(rows).round(4)
    base_npv = scenario_df.loc[scenario_df["Scenario"] == "Base", "NPV Mean"].values[0]
    scenario_df["NPV_Delta_vs_Base"] = scenario_df["NPV Mean"] - base_npv
    return scenario_df, macro_df, macro_hist_scenarios, macro_source, macro_base_rate


def build_discount_rate_sensitivity(base_config, base_df):
    macro_cfg, macro_df, macro_hist_scenarios, macro_source, macro_base_rate = build_macro_base_config(base_config)

    grid = [
        max(0.01, macro_base_rate - 0.01),
        macro_base_rate,
        macro_base_rate + 0.01,
        macro_base_rate + 0.02,
    ]

    rows = []
    quick_cfg = make_secondary_config(macro_cfg)

    for dr in grid:
        cfg = quick_cfg.copy()
        cfg["discount_rate"] = dr
        cfg["valuation_discount_rate"] = dr
        cfg["hurdle_rate"] = base_config["hurdle_rate"]
        cfg["scenario_name"] = f"Valuation_{dr:.2%}"

        res = run_sim(cfg, base_df)

        rows.append({
            "Valuation Discount Rate": dr,
            "IRR Mean": res["monte_carlo"]["irr_mean"],
            "MOIC Mean": res["monte_carlo"]["moic_mean"],
            "NPV Mean": res["monte_carlo"]["npv_mean"],
            "NPV CVaR (5%)": res["risk"]["npv_cvar_5"],
            "Prob(NPV<0)": res["risk"]["prob_npv_negative"],
        })

    return pd.DataFrame(rows).round(4)


def build_driver_table(scenario_df, discount_df):
    rows = []

    discount_impact = discount_df["NPV Mean"].max() - discount_df["NPV Mean"].min()
    rows.append({
        "Driver": "Valuation Discount Rate",
        "Why it matters": "Required return assumptions directly shift valuation levels and investability.",
        "Impact Score": abs(discount_impact),
    })

    scenario_impact = scenario_df["NPV Mean"].max() - scenario_df["NPV Mean"].min()
    rows.append({
        "Driver": "Scenario Layer",
        "Why it matters": "Joint scenario assumptions around valuation rate, exit multiple and volatility drive overall robustness.",
        "Impact Score": abs(scenario_impact),
    })

    rows.append({
        "Driver": "Contract Structure",
        "Why it matters": "Value and downside can change materially with payout timing, fee structure and structural mechanics.",
        "Impact Score": 6.79,
    })

    driver_df = pd.DataFrame(rows)
    driver_df = driver_df.sort_values("Impact Score", ascending=False).reset_index(drop=True)
    driver_df["Rank"] = np.arange(1, len(driver_df) + 1)
    driver_df = driver_df[["Rank", "Driver", "Why it matters", "Impact Score"]].head(3)

    return driver_df

# =========================
# Run app
# =========================
if run_button:
    with st.spinner("Running valuation, underwriting checks, scenario analysis and sensitivity views..."):
        entry_ev = initial_cashflow * entry_multiple
        entry_equity = entry_ev - entry_debt
        implied_entry_ltv = entry_debt / entry_ev if entry_ev > 0 else np.nan
        illustrative_gross_target_return = (
            (hurdle_rate + fund_mgmt_fee) / (1 - fund_carry)
            if (1 - fund_carry) != 0 else np.nan
        )

        plausibility_warnings = build_plausibility_warnings(
            entry_multiple=entry_multiple,
            valuation_discount_rate=valuation_discount_rate,
            hurdle_rate=hurdle_rate,
            ltv_target=ltv_target,
            interest_rate=interest_rate,
            cashflow_volatility=cashflow_volatility,
        )

        base_config = {
            "preset_name": "Reference Defaults / Manual Override",
            "initial_cashflow": initial_cashflow,
            "growth_rate": growth_rate,
            "contract_length": contract_length,
            "entry_multiple": entry_multiple,
            "exit_multiple": exit_multiple,
            "multiple_sensitivity": multiple_sensitivity,
            "valuation_discount_rate": valuation_discount_rate,
            "hurdle_rate": hurdle_rate,
            "entry_debt": entry_debt,
            "ltv_target": ltv_target,
            "entry_ev": entry_ev,
            "entry_equity": entry_equity,
            "implied_entry_ltv": implied_entry_ltv,
            "operating_fee": operating_fee,
            "consortium_fee": consortium_fee,
            "n_simulations": int(n_simulations),
            "sigma_cf": cashflow_volatility,
            "investor_share": investor_share,
            "interest_rate": interest_rate,
            "use_tail_overlay": use_tail_overlay,
            "tail_event_prob": tail_event_prob,
            "tail_event_severity": tail_event_severity,
            "super_tail_prob": super_tail_prob,
            "super_tail_severity": super_tail_severity,
            "fund_mgmt_fee": fund_mgmt_fee,
            "fund_carry": fund_carry,
            "illustrative_gross_target_return": illustrative_gross_target_return,
            "use_fred": use_fred,
            "macro_lookback_months": macro_lookback_months,
        }

        macro_base_config, macro_df, macro_hist_scenarios, macro_source, macro_base_rate = build_macro_base_config(base_config)

        fed_funds = macro_base_config.get("fed_funds")
        fed_display = f"{fed_funds:.2f}%" if fed_funds is not None else "n/a"

        results = run_sim(macro_base_config, reference_clean)
        det = results["deterministic"]
        mc = results["monte_carlo"]
        risk = results["risk"]
        mc_raw = results["raw_mc"]
        overlay_stats = results["tail_overlay"]

        decision = make_decision(mc, risk, macro_base_config["hurdle_rate"])
        run_id = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        scenario_df, macro_df, macro_hist_scenarios, macro_source, macro_base_rate = build_scenario_table(
            base_config,
            reference_clean,
            base_mc=mc,
            base_risk=risk,
            base_decision=decision,
        )
        discount_df = build_discount_rate_sensitivity(base_config, reference_clean)
        driver_df = build_driver_table(scenario_df, discount_df)

        primary_driver = driver_df.iloc[0]["Driver"] if not driver_df.empty else "Valuation Discount Rate"
        underwriting_reasons = build_underwriting_reasons(
            mc,
            risk,
            macro_base_config["hurdle_rate"],
            primary_driver,
        )

        underwriting_reasons = underwriting_reasons[:4]

        positive_points, caution_points = build_why_invest_lists(
            mc,
            risk,
            decision,
            primary_driver,
            macro_base_config["hurdle_rate"],
        )

        worst_case = scenario_df.loc[scenario_df["NPV Mean"].idxmin()]
        caution_points.append(
            f"Worst-case scenario ({worst_case['Scenario']}): NPV {worst_case['NPV Mean']:.2f}, Prob(NPV<0) {worst_case['Prob(NPV<0)']:.2%}."
        )

        positive_points = positive_points[:3]
        caution_points = caution_points[:3]
        
        run_metadata = {
            "preset": "Reference Defaults / Manual Override",
            "run_timestamp_utc": datetime.utcnow().isoformat(),
            "inputs": base_config,
            "decision": decision,
            "risk": risk,
        }

        meta_info = {
            "Run_ID": run_id,
            "IRR_Mean_Base": mc["irr_mean"],
            "Prob_NPV_Neg_Base": risk["prob_npv_negative"],
            "MC_Runs": base_config.get("n_simulations", 3000),
            "Discount_Rate_Base": macro_base_rate,
        }

        for k, v in meta_info.items():
            scenario_df[k] = v

    tab1, tab2, tab3 = st.tabs(["Overview", "Scenarios", "Downloads"])

    with tab1:
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

        st.subheader("IC Snapshot")

        spread = decision.get("Hurdle_Spread", np.nan)
        if pd.isna(spread):
            spread_label = "n/a"
        elif spread > 0.03:
            spread_label = f"{spread:+.2%} (STRONG BUFFER)"
        elif spread > 0.01:
            spread_label = f"{spread:+.2%} (COMFORTABLE)"
        elif spread > 0:
            spread_label = f"{spread:+.2%} (TIGHT)"
        else:
            spread_label = f"{spread:+.2%} (BELOW HURDLE)"

        
        snap1, snap2, snap3, snap4 = st.columns(4)
        with snap1:
            st.metric("Recommendation", decision["FINAL_DECISION"])
        with snap2:
            st.metric("Risk Flag", decision["Risk_Flag"])
        with snap3:
            st.metric("Total Score", int(decision["Total_Score"]))
        with snap4:
            st.metric("Hurdle Spread", spread_label)

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.subheader("Why Invest / Why Not")

        why1, why2 = st.columns(2)

        with why1:
            st.markdown("**Investment merits**")
            if positive_points:
                for p in positive_points:
                    st.markdown(f"- {p}")
            else:
                st.caption("No major strengths identified under the current assumptions.")

        with why2:
            st.markdown("**Key watchpoints**")
            if caution_points:
                for p in caution_points:
                    st.markdown(f"- {p}")
            else:
                st.caption("No major caution flags identified under the current assumptions.")

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.subheader("Macro Context")

        st.markdown(
            f"""
            <div class="method-note">
            <b>Macro source:</b> {macro_source} &nbsp;&nbsp;&nbsp;
            <b>Fed Funds:</b> {fed_display} &nbsp;&nbsp;&nbsp;
            <b>Macro-implied base valuation rate:</b> {macro_base_rate:.2%} &nbsp;&nbsp;&nbsp;
            <b>Hurdle rate:</b> {macro_base_config["hurdle_rate"]:.2%}
            </div>
            """,
            unsafe_allow_html=True,
        )

        fed_decimal = (fed_funds / 100) if fed_funds is not None else np.nan
        spread_vs_rf = macro_base_rate - fed_decimal if not pd.isna(fed_decimal) else np.nan

        if not pd.isna(spread_vs_rf):
            st.markdown(
                f"""
                <div style="font-size:12px; color:#6b7280; margin-top:4px;">
                Implied risk premium: {spread_vs_rf:.2%}
                </div>
                """,
                unsafe_allow_html=True,
            )

       
        st.markdown(f"""
            <div style="font-size:12px; color:#6b7280; margin-top:6px;">
            Run ID: {run_id} | Base Case | IRR: {mc["irr_mean"]:.2%} | Prob(NPV&lt;0): {risk["prob_npv_negative"]:.2%} | MC: {base_config.get("n_simulations", 3000)} | r: {macro_base_rate:.2%}
            </div>
            """, unsafe_allow_html=True)

      
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.subheader("Decision Scorecard")

        st.caption(
            "Scoring combines return (IRR, MOIC, NPV), risk (downside probability, CVaR) and resilience (P10 IRR, hurdle buffer)."
        )

        score1, score2, score3, score4 = st.columns(4)
        with score1:
            st.metric("Return Score", decision["Return_Score"])
        with score2:
            st.metric("Risk Score", decision["Risk_Score"])
        with score3:
            st.metric("Resilience Score", decision.get("Resilience_Score", "n/a"))
        with score4:
            st.metric("Hard Gates Passed", "Yes" if decision.get("Hard_Gates_Passed", False) else "No")
        
        if plausibility_warnings:
            st.markdown(
                "<div class='warning-box'><b>Plausibility warnings</b><br>"
                + "<br>".join([f"• {w}" for w in plausibility_warnings])
                + "</div>",
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.subheader("IC Interpretation")

        interpretation_text = decision.get("Interpretation", "No interpretation available.")

        if primary_driver == "Scenario Layer":
            interpretation_text += " Primary value driver is scenario layer, meaning robustness depends on coordinated shifts in rates, multiples and volatility."
        elif primary_driver == "Valuation Discount Rate":
            interpretation_text += " Primary value driver is valuation discount rate, indicating high sensitivity to required return assumptions."
        else:
            interpretation_text += " Primary value driver is contract structure, meaning value and downside protection are shaped by structural deal mechanics."

        if decision["FINAL_DECISION"] == "INVEST":
            st.success(
                f"Recommendation: {decision['FINAL_DECISION']}\n\n"
                f"{interpretation_text}"
            )
        elif decision["FINAL_DECISION"] == "INVEST WITH CONDITIONS":
            st.warning(
                f"Recommendation: {decision['FINAL_DECISION']}\n\n"
                f"{interpretation_text}"
            )
        else:
            st.error(
                f"Recommendation: {decision['FINAL_DECISION']}\n\n"
                f"{interpretation_text}"
            )



        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("## Investment Decision Summary")

        decision_class = {
            "INVEST": "invest",
            "INVEST WITH CONDITIONS": "",
            "REJECT": "reject",
        }[decision["FINAL_DECISION"]]

        st.markdown(
            f"""
            <div class="decision-summary-card {decision_class}">
                <div class="decision-title">Final decision</div>
                <div class="decision-main">{decision["FINAL_DECISION"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        s1, s2, s3 = st.columns(3)
        with s1:
            render_risk_flag_box(decision["Risk_Flag"])
        with s2:
            st.metric("Hard Gates Passed", "Yes" if decision["Hard_Gates_Passed"] else "No")
        with s3:
            st.metric("Hurdle Spread", f"{(mc['irr_mean'] - macro_base_config['hurdle_rate']):+.2%}")

        st.caption("Hard gates are necessary but not sufficient; final classification also reflects the combined return/risk score.")

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### Underwriting Rationale")
        for reason in underwriting_reasons:
            st.markdown(f"- {reason}")

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### Decision Gates")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("IRR ≥ Hurdle Rate", "Pass" if decision["Gate_IRR"] else "Fail")
        g2.metric("MOIC ≥ 2.0x", "Pass" if decision["Gate_MOIC"] else "Fail")
        g3.metric("Prob(NPV<0) < 40%", "Pass" if decision["Gate_ProbNeg"] else "Fail")
        g4.metric("NPV CVaR > -20", "Pass" if decision["Gate_CVaR"] else "Fail")

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### Reference Defaults")
        st.markdown(
            """
            <span class="info-chip">Reference defaults loaded</span>
            <span class="info-chip">Anonymised setup</span>
            <span class="info-chip">Standalone decision tool</span>
            """,
            unsafe_allow_html=True,
        )

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Investor Share", fmt_pct(REFERENCE_INVESTOR_SHARE))
        b2.metric("Reference Equity Ticket", fmt_num(REFERENCE_EQUITY_TICKET))
        b3.metric("Reference Entry EV", fmt_num(REFERENCE_ENTRY_EV))
        b4.metric("MC Runs", fmt_int(n_simulations))

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### Model Framework")
        st.markdown(
            """
            <div class="method-note">
            <b>Framework.</b> Deterministic reference metrics act as the calibration anchor.<br>
            Monte Carlo extends the same cashflow base with uncertainty in revenue, exit multiple and valuation-rate assumptions.<br>
            The tool is designed as a standalone investor-level underwriting and decision system.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Deterministic Reference vs Monte Carlo Extension")
        c_ref1, c_ref2, c_ref3, c_ref4 = st.columns(4)
        c_ref1.metric("Deterministic IRR", fmt_pct(det["IRR"]))
        c_ref2.metric("Deterministic MOIC", fmt_x(det["MOIC"]))
        c_ref3.metric("Valuation Discount Rate", fmt_pct(macro_base_rate))
        c_ref4.metric("Hurdle Rate", fmt_pct(macro_base_config["hurdle_rate"]))

        st.markdown(
            """
            <div class="method-note">
            <b>Method note.</b> Displayed IRR / MOIC metrics reflect modeled investor-level cashflows in the current setup.
            Interpretation depends on the modeled fee layer and return definition applied to the underlying cashflow series.
            Detailed outputs are available in the Downloads tab.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.subheader("Risk & Distribution")
        c1, c2 = st.columns(2)

        with c1:
            irr_annotation = (
                f"Mean: {mc['irr_mean']:.2%}\n"
                f"P10: {mc['irr_p10']:.2%}\n"
                f"P90: {mc['irr_p90']:.2%}"
            )
            fig_irr = build_clean_hist(
                mc_raw["irr"],
                title="IRR Distribution",
                xlabel="IRR",
                threshold=hurdle_rate,
                threshold_color=GREEN,
                mean_line=np.nanmean(mc_raw["irr"]),
                annotation=irr_annotation,
            )
            fig_irr.axes[0].xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            st.pyplot(fig_irr, use_container_width=True)

        with c2:
            npv_annotation = (
                f"Mean: {mc['npv_mean']:.2f}\n"
                f"P10: {mc['npv_p10']:.2f}\n"
                f"P90: {mc['npv_p90']:.2f}\n"
                f"Prob(NPV<0): {risk['prob_npv_negative']:.2%}"
            )
            fig_npv = build_clean_hist(
                mc_raw["npv"],
                title="NPV Distribution",
                xlabel="NPV",
                threshold=0,
                threshold_color=BLUE,
                mean_line=np.nanmean(mc_raw["npv"]),
                annotation=npv_annotation,
            )
            fig_npv.axes[0].xaxis.set_major_formatter(FuncFormatter(money_formatter))
            st.pyplot(fig_npv, use_container_width=True)

        st.caption("Dashed line = threshold; dotted line = simulated mean.")

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.subheader("Downside & Fund-Level View")
        r1, r2, r3 = st.columns(3)
        r1.metric("Prob. Negative NPV", fmt_pct(risk["prob_npv_negative"]))
        r2.metric("NPV CVaR (5%)", fmt_num(risk["npv_cvar_5"]))
        r3.metric("Valuation Discount Rate", fmt_pct(macro_base_rate))

        f1, f2, f3 = st.columns(3)
        f1.metric("Hurdle Spread", f"{(mc['irr_mean'] - hurdle_rate):+.2%}")
        f2.metric("Mgmt Fee Drag", fmt_pct(fund_mgmt_fee))
        f3.metric("Illustrative Gross Target", fmt_pct(illustrative_gross_target_return))

        st.caption(
            "Illustrative bridge only. Not a full fund waterfall. Valuation discount rate and hurdle rate are shown separately to distinguish valuation from decision logic."
        )

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.subheader("Summary Tables")
        det_df = pd.DataFrame([det]).round(4)
        mc_df = pd.DataFrame([{
            "IRR Mean": mc["irr_mean"],
            "IRR P10": mc["irr_p10"],
            "IRR P90": mc["irr_p90"],
            "MOIC Mean": mc["moic_mean"],
            "MOIC P10": mc["moic_p10"],
            "MOIC P90": mc["moic_p90"],
            "NPV Mean": mc["npv_mean"],
            "NPV P10": mc["npv_p10"],
            "NPV P90": mc["npv_p90"],
            "Prob(NPV<0)": risk["prob_npv_negative"],
            "NPV CVaR (5%)": risk["npv_cvar_5"],
        }]).round(4)

        st.markdown("**Deterministic Reference Summary**")
        render_clean_table(format_det_display(det_df))

        st.markdown("**Monte Carlo Highlights**")
        render_clean_table(format_mc_display(mc_df))

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("## Top Value Drivers")
        max_impact = driver_df["Impact Score"].max() if not driver_df.empty else 1.0

        for _, row in driver_df.iterrows():
            impact_ratio = 0 if max_impact == 0 else min(row["Impact Score"] / max_impact, 1.0)
            fill_pct = impact_ratio * 100

            driver_name = row["Driver"]

            if driver_name == "Valuation Discount Rate":
                explanation = "Primary sensitivity driver — small changes materially impact NPV."
            elif driver_name == "Scenario Layer":
                explanation = "Macro + multiple + volatility jointly define robustness."
            else:
                explanation = "Structural contract features shape downside protection."

            st.markdown(
                f"""
                <div class="driver-row">
                    <div class="driver-main">
                        <div class="driver-rank">#{int(row['Rank'])}</div>
                        <div class="driver-title">{row['Driver']}</div>
                        <div class="driver-desc">{row['Why it matters']}</div>
                        <div class="driver-desc" style="margin-top:6px; color:#64748b;">{explanation}</div>
                    </div>
                    <div class="driver-side">
                        <div class="driver-side-label">Impact Score</div>
                        <div class="driver-side-value">{row['Impact Score']:.2f}</div>
                        <div class="driver-bar-wrap">
                            <div class="driver-bar-fill" style="width:{fill_pct:.1f}%"></div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.subheader("Optional Stylised Downside Overlay")

        if use_tail_overlay:
            o1, o2 = st.columns(2)
            o1.metric("Tail Events Triggered", fmt_int(overlay_stats["tail_event_count"]))
            o2.metric("Super-Tail Events Triggered", fmt_int(overlay_stats["super_tail_event_count"]))
            st.caption("Enabled as a prototype extension. Primary downside still comes from volatility and stressed scenarios.")
        else:
            st.info("Overlay disabled. Primary downside is currently captured through Monte Carlo volatility, scenario stress and valuation sensitivity.")

    with tab2:
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.subheader("Scenario / Robustness View")
        st.markdown(
            f"""
            <div class="method-note">
                <b>Macro source:</b> {macro_source} &nbsp;&nbsp;&nbsp;&nbsp;
                <b>Macro-implied base valuation rate:</b> {macro_base_rate:.2%} &nbsp;&nbsp;&nbsp;&nbsp;
                <b>Fed Funds:</b> {fed_display}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="scenario-note">Scenario cards summarise underwriting outcomes under coordinated shifts in valuation rate, exit multiple and cashflow volatility.</div>',
            unsafe_allow_html=True,
        )

        cols = st.columns(4)
        rows_list = list(scenario_df.iterrows())
        for idx, col in enumerate(cols):
            with col:
                if idx < len(rows_list):
                    _, row = rows_list[idx]
                    render_scenario_card(row, base_irr=mc["irr_mean"])

        st.markdown("---")
        st.markdown("### Scenario Comparison Table")
        scenario_table_display = scenario_table_for_display(
            scenario_df[[
                "Scenario",
                "Valuation Discount Rate",
                "Hurdle Rate",
                "Exit Multiple",
                "Volatility",
                "IRR Mean",
                "MOIC Mean",
                "NPV Mean",
                "Prob(NPV<0)",
                "NPV CVaR (5%)",
                "Decision",
                "Risk Flag",
            ]]
        )
        render_clean_table(
            scenario_table_display,
            decision_col="Decision",
            risk_col="Risk Flag",
        )

        worst_case = scenario_df.loc[scenario_df["NPV Mean"].idxmin()]

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="method-note" style="border-left: 4px solid {RED}; background: {RED_SOFT};">
                <b>Worst-case scenario:</b> {worst_case['Scenario']} &nbsp;&nbsp;&nbsp;
                <b>NPV Mean:</b> {worst_case['NPV Mean']:.2f} &nbsp;&nbsp;&nbsp;
                <b>Prob(NPV&lt;0):</b> {worst_case['Prob(NPV<0)']:.2%} &nbsp;&nbsp;&nbsp;
                <b>Decision:</b> {worst_case['Decision']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### Scenario Charts")
        sc1, sc2 = st.columns(2)

        with sc1:
            fig_s1, ax_s1 = plt.subplots(figsize=(5.8, 3.0))
            x = np.arange(len(scenario_df))
            ax_s1.bar(x, scenario_df["NPV Mean"], alpha=0.88, color=BLUE)
            ax_s1.axhline(0, linestyle="--", linewidth=1.5, color=BLUE)
            ax_s1.set_xticks(x)
            ax_s1.set_xticklabels(scenario_df["Scenario"])
            ax_s1.set_title("NPV Mean by Scenario", fontsize=11, pad=8)
            ax_s1.set_ylabel("NPV", fontsize=9.5)
            ax_s1.grid(alpha=0.08, linewidth=0.55)
            ax_s1.tick_params(axis="both", labelsize=8.8)
            ax_s1.yaxis.set_major_formatter(FuncFormatter(money_formatter))
            fig_s1.tight_layout()
            st.pyplot(fig_s1, use_container_width=False)

        with sc2:
            fig_s2, ax_s2 = plt.subplots(figsize=(5.8, 3.0))
            x = np.arange(len(scenario_df))
            ax_s2.plot(x, scenario_df["Prob(NPV<0)"], marker="o", linewidth=1.8, color=BLUE)
            ax_s2.set_xticks(x)
            ax_s2.set_xticklabels(scenario_df["Scenario"])
            ax_s2.set_title("Downside Probability by Scenario", fontsize=11, pad=8)
            ax_s2.set_ylabel("Prob(NPV<0)", fontsize=9.5)
            ax_s2.set_ylim(0, 1.0)
            ax_s2.grid(alpha=0.08, linewidth=0.55)
            ax_s2.tick_params(axis="both", labelsize=8.8)
            ax_s2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            fig_s2.tight_layout()
            st.pyplot(fig_s2, use_container_width=False)

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### Valuation Discount Rate Sensitivity")
        render_clean_table(discount_table_for_display(discount_df))

        fig_tr, ax_tr = plt.subplots(figsize=(6.2, 3.2))
        ax_tr.plot(discount_df["Valuation Discount Rate"], discount_df["NPV Mean"], marker="o", linewidth=1.8, color=BLUE)
        ax_tr.axhline(0, linestyle="--", linewidth=1.5, color=BLUE)
        ax_tr.text(
            0.985,
            0.08,
            "NPV = 0 break-even",
            transform=ax_tr.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            color=BLUE
        )
        ax_tr.set_title("Valuation Discount Rate Sensitivity", fontsize=11, pad=8)
        ax_tr.set_xlabel("Valuation Discount Rate", fontsize=9.5)
        ax_tr.set_ylabel("NPV Mean", fontsize=9.5)
        ax_tr.grid(alpha=0.08, linewidth=0.55)
        ax_tr.tick_params(axis="both", labelsize=8.8)
        ax_tr.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax_tr.yaxis.set_major_formatter(FuncFormatter(money_formatter))
        fig_tr.tight_layout()

        center_left, center_mid, center_right = st.columns([1, 2.2, 1])
        with center_mid:
            st.pyplot(fig_tr, use_container_width=False)

        st.caption("IRR is independent of the valuation discount rate. This sensitivity isolates valuation impact through NPV and downside metrics.")

    with tab3:
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.subheader("Download Outputs")
        st.caption("Detailed cashflow series, scenario tables and IC-style summaries can be used as an evidence pack for memo / committee materials.")

        decision_df = pd.DataFrame([decision]).round(4)
        risk_df = pd.DataFrame([risk]).round(4)
        mc_export_df = pd.DataFrame([{
            "IRR Mean": mc["irr_mean"],
            "IRR P10": mc["irr_p10"],
            "IRR P90": mc["irr_p90"],
            "MOIC Mean": mc["moic_mean"],
            "MOIC P10": mc["moic_p10"],
            "MOIC P90": mc["moic_p90"],
            "NPV Mean": mc["npv_mean"],
            "NPV P10": mc["npv_p10"],
            "NPV P90": mc["npv_p90"],
            "Prob(NPV<0)": risk["prob_npv_negative"],
            "NPV CVaR (5%)": risk["npv_cvar_5"],
        }]).round(4)

        d1, d2, d3, d4, d5 = st.columns(5)
        with d1:
            st.download_button("Decision CSV", df_to_csv_bytes(decision_df), "decision_summary.csv", "text/csv")
        with d2:
            st.download_button("Monte Carlo CSV", df_to_csv_bytes(mc_export_df), "monte_carlo_highlights.csv", "text/csv")
        with d3:
            st.download_button("Risk CSV", df_to_csv_bytes(risk_df), "risk_metrics.csv", "text/csv")
        with d4:
            det_export_df = pd.DataFrame([det]).round(4)
            st.download_button("Deterministic CSV", df_to_csv_bytes(det_export_df), "deterministic_reference.csv", "text/csv")
        with d5:
            st.download_button("Run Metadata JSON", json_to_bytes(run_metadata), "run_metadata.json", "application/json")

        x1, x2, x3, x4 = st.columns(4)
        with x1:
            st.download_button("Scenario Comparison CSV", df_to_csv_bytes(scenario_df), "scenario_comparison.csv", "text/csv")
        with x2:
            st.download_button("Valuation Rate Sensitivity CSV", df_to_csv_bytes(discount_df), "valuation_rate_sensitivity.csv", "text/csv")
        with x3:
            st.download_button("Top Drivers CSV", df_to_csv_bytes(driver_df), "top_value_drivers.csv", "text/csv")
    with x4:
            ic_summary_text = build_ic_summary_text(
                decision=decision,
                mc=mc,
                risk=risk,
                macro_base_rate=macro_base_rate,
                hurdle_rate_used=macro_base_config["hurdle_rate"],
                macro_source=macro_source,
                fed_display=fed_display,
                underwriting_reasons=underwriting_reasons,
            )
            st.download_button(
                "IC Summary TXT",
                ic_summary_text.encode("utf-8"),
                "ic_summary.txt",
                "text/plain"
            )

else:
    st.info("Adjust the model inputs in the sidebar and click “Run Valuation”.")