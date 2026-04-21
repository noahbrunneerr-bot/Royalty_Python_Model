import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
from src.mc import run_pg3_monte_carlo, summarize_pg3_mc

# =========================
# Page config
# =========================
st.set_page_config(page_title="Royalty Valuation Tool", layout="wide")

st.title("Royalty Valuation Tool")
st.caption("Investor-oriented royalty valuation, scenario analysis, and IC decision support.")

# =========================
# Load ground truth / base case data
# =========================
gt_path = project_root / "Data" / "processed" / "ground_truth_clean.csv"
gt_df = pd.read_csv(gt_path)
gt_clean = gt_df[gt_df["Net_CF_to_Consortium"].notna()].copy()

fy = gt_clean["FY"].to_numpy()
net_cf = gt_clean["Net_CF_to_Consortium"].to_numpy(dtype=float)
interest_rate_arr = gt_clean["Interest_Rate"].to_numpy(dtype=float)
nav_multiple = gt_clean["NAV_Multiple"].to_numpy(dtype=float)
mandatory_amort = gt_clean["Mandatory_Amort"].to_numpy(dtype=float)
consortium_fees_arr = gt_clean["Consortium_Fees"].to_numpy(dtype=float)

series_pg3 = ScenarioSeries(
    fy=fy,
    net_cf_to_consortium=net_cf,
    interest_rate=interest_rate_arr,
    nav_multiple=nav_multiple,
    mandatory_amort=mandatory_amort,
    consortium_fees=consortium_fees_arr,
)

# =========================
# Sidebar inputs
# =========================
st.sidebar.header("Core Deal Inputs")

initial_cashflow = st.sidebar.number_input("Initial Cashflow", value=50_000_000.0, step=1_000_000.0)
growth_rate = st.sidebar.number_input("Growth Rate", value=0.03, step=0.005, format="%.3f")
contract_length = st.sidebar.number_input("Contract Length", value=15, step=1)

entry_multiple = st.sidebar.number_input("Entry Multiple", value=13.9, step=0.1)
exit_multiple = st.sidebar.number_input("Exit Multiple", value=12.0, step=0.1)

entry_debt = st.sidebar.number_input("Entry Debt", value=325_000_000.0, step=5_000_000.0)
ltv_target = st.sidebar.number_input("LTV Target", value=0.50, step=0.01, format="%.2f")

operating_fee = st.sidebar.number_input("Operating Fee", value=0.05, step=0.005, format="%.3f")
consortium_fee = st.sidebar.number_input("Consortium Fee", value=0.0008, step=0.0001, format="%.4f")

investor_target_return = st.sidebar.number_input(
    "Investor Target Return", value=0.10, step=0.005, format="%.3f"
)
cashflow_volatility = st.sidebar.number_input(
    "Cashflow Volatility", value=0.15, step=0.01, format="%.2f"
)
n_simulations = st.sidebar.number_input("Monte Carlo Runs", value=5000, step=500)

investor_share = st.sidebar.number_input("Investor Share", value=0.2954, step=0.01, format="%.4f")
interest_rate = st.sidebar.number_input("Interest Rate", value=0.06, step=0.005, format="%.3f")

st.sidebar.markdown("---")
st.sidebar.header("Scenario Controls")

upside_rate_delta = st.sidebar.number_input("Upside: Target Return Δ", value=-0.01, step=0.005, format="%.3f")
upside_exit_delta = st.sidebar.number_input("Upside: Exit Multiple Δ", value=0.5, step=0.1)
upside_vol_delta = st.sidebar.number_input("Upside: Volatility Δ", value=-0.02, step=0.01, format="%.2f")

high_rate_delta = st.sidebar.number_input("High Rate: Target Return Δ", value=0.02, step=0.005, format="%.3f")
high_rate_exit_delta = st.sidebar.number_input("High Rate: Exit Multiple Δ", value=-1.0, step=0.1)
high_rate_vol_delta = st.sidebar.number_input("High Rate: Volatility Δ", value=0.03, step=0.01, format="%.2f")

stress_rate_delta = st.sidebar.number_input("Stress: Target Return Δ", value=0.03, step=0.005, format="%.3f")
stress_exit_delta = st.sidebar.number_input("Stress: Exit Multiple Δ", value=-1.5, step=0.1)
stress_vol_delta = st.sidebar.number_input("Stress: Volatility Δ", value=0.05, step=0.01, format="%.2f")

st.sidebar.markdown("---")
st.sidebar.header("Tail / Super-Tail Overlay")

tail_event_prob = st.sidebar.number_input("Tail Event Probability", value=0.03, step=0.01, format="%.2f")
tail_event_severity = st.sidebar.number_input("Tail Event Severity", value=0.20, step=0.05, format="%.2f")

super_tail_prob = st.sidebar.number_input("Super-Tail Probability", value=0.01, step=0.005, format="%.3f")
super_tail_severity = st.sidebar.number_input("Super-Tail Severity", value=0.40, step=0.05, format="%.2f")

st.sidebar.markdown("---")
st.sidebar.header("Illustrative Fund-Level Bridge")

fund_mgmt_fee = st.sidebar.number_input("Mgmt Fee Drag", value=0.02, step=0.005, format="%.3f")
fund_carry = st.sidebar.number_input("Carry Rate", value=0.20, step=0.05, format="%.2f")

run_button = st.sidebar.button("Run Valuation")

# =========================
# Helper functions
# =========================
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def safe_clip(x, low=None, high=None):
    if low is not None:
        x = max(x, low)
    if high is not None:
        x = min(x, high)
    return x


def apply_tail_overlay(mc_out, tail_prob, tail_severity, super_tail_prob, super_tail_severity):
    rng = np.random.default_rng(123)
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
    NSIM = config["n_simulations"]
    SIGMA_CF = config["sigma_cf"]
    SIGMA_MULTIPLE = config.get("sigma_multiple", 0.20)
    TARGET_RETURN = config["investor_target_return"]

    EXIT_MULTIPLE = config["exit_multiple"]
    OPERATING_FEE = config["operating_fee"]
    CONSORTIUM_FEE = config["consortium_fee"]
    INTEREST_RATE = config["interest_rate"]

    ENTRY_EV = config["entry_ev"]
    ENTRY_DEBT = config["entry_debt"]
    INVESTOR_SHARE = config["investor_share"]
    LTV_TARGET = config["ltv_target"]

    UNIT_SCALE = 1_000_000

    entry_ev_tmp = ENTRY_EV / UNIT_SCALE
    entry_debt_tmp = ENTRY_DEBT / UNIT_SCALE
    entry_equity_total_tmp = entry_ev_tmp - entry_debt_tmp
    entry_equity_investor_tmp = entry_equity_total_tmp * INVESTOR_SHARE

    params_tmp = WaterfallParams(
        scenario=config.get("scenario_name", "Base"),
        entry_debt=entry_debt_tmp,
        recap_target_ltv=LTV_TARGET,
        recap_years=(fy[3], fy[6]),
        cash_sweep_start_fy=fy[2],
        operating_fee_pct=OPERATING_FEE,
        pg_share=INVESTOR_SHARE,  # internal field kept for compatibility
        consortium_equity_ticket=entry_equity_investor_tmp,
    )

    wf_tmp = compute_waterfall(params_tmp, series_pg3)
    outs_tmp = summarize_outputs(params_tmp, wf_tmp)

    det_summary = {
        "IRR": outs_tmp["IRR (annual)"],
        "MOIC": outs_tmp["MOIC"],
    }

    mc_out = run_pg3_monte_carlo(
        base_df=base_df,
        exit_multiple_mean=EXIT_MULTIPLE,
        exit_multiple_sigma=SIGMA_MULTIPLE,
        n_sim=NSIM,
        random_state=42,
        discount_rate=TARGET_RETURN,
        sigma_cf=SIGMA_CF,
        operating_fee_rate=OPERATING_FEE,
        consortium_fee_rate=CONSORTIUM_FEE,
        interest_rate=INTEREST_RATE,
        amort_rate=0.0,
        cash_sweep_rate=0.0,
        recap_year=None,
        recap_amount=0.0,
    )

    BASE_ENTRY_DEBT = 325_000_000
    BASE_ENTRY_EV = 50_000_000 * 13.9

    base_equity = BASE_ENTRY_EV - BASE_ENTRY_DEBT
    current_equity = ENTRY_EV - ENTRY_DEBT
    equity_scale = current_equity / base_equity if base_equity > 0 else 1.0

    mc_out["npv"] = mc_out["npv"] * equity_scale
    mc_out["moic"] = mc_out["moic"] * equity_scale

    leverage_ratio = ENTRY_DEBT / ENTRY_EV if ENTRY_EV > 0 else 0
    base_leverage_ratio = BASE_ENTRY_DEBT / BASE_ENTRY_EV if BASE_ENTRY_EV > 0 else 0
    irr_shift = (leverage_ratio - base_leverage_ratio) * 0.02
    mc_out["irr"] = mc_out["irr"] + irr_shift

    mc_out, overlay_stats = apply_tail_overlay(
        mc_out,
        tail_prob=config.get("tail_event_prob", 0.0),
        tail_severity=config.get("tail_event_severity", 0.0),
        super_tail_prob=config.get("super_tail_prob", 0.0),
        super_tail_severity=config.get("super_tail_severity", 0.0),
    )

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
        "mc_raw": mc_out,
        "overlay_stats": overlay_stats,
    }


def make_decision(mc, risk, investor_target_return):
    irr = mc.get("irr_mean", np.nan)
    moic = mc.get("moic_mean", np.nan)
    npv = mc.get("npv_mean", np.nan)
    prob_neg = risk.get("prob_npv_negative", np.nan)
    npv_cvar = risk.get("npv_cvar_5", np.nan)

    gate_irr_ok = irr >= investor_target_return
    gate_moic_ok = moic >= 2.0
    gate_prob_ok = prob_neg <= 0.40
    gate_cvar_ok = npv_cvar > -20

    hard_gate_pass = all([gate_irr_ok, gate_moic_ok, gate_prob_ok, gate_cvar_ok])

    return_score = 0
    if irr >= investor_target_return + 0.02:
        return_score += 3
    elif irr >= investor_target_return:
        return_score += 2
    elif irr >= investor_target_return - 0.01:
        return_score += 1

    if moic >= 2.5:
        return_score += 2
    elif moic >= 2.0:
        return_score += 1

    if npv > 0:
        return_score += 1

    risk_score = 0
    if prob_neg <= 0.20:
        risk_score += 2
    elif prob_neg <= 0.40:
        risk_score += 1

    if npv_cvar > -10:
        risk_score += 2
    elif npv_cvar > -20:
        risk_score += 1

    total_score = return_score + risk_score

    if hard_gate_pass and total_score >= 6 and prob_neg <= 0.30:
        final_decision = "INVEST"
    elif total_score >= 4:
        final_decision = "INVEST WITH CONDITIONS"
    else:
        final_decision = "REJECT"

    if prob_neg > 0.50:
        risk_flag = "HIGH DOWNSIDE RISK"
    elif prob_neg > 0.35:
        risk_flag = "ELEVATED DOWNSIDE RISK"
    elif prob_neg > 0.20:
        risk_flag = "MODERATE DOWNSIDE RISK"
    else:
        risk_flag = "ACCEPTABLE DOWNSIDE RISK"

    if final_decision == "INVEST":
        interpretation = (
            "Attractive investor-oriented risk-return profile. "
            "IRR exceeds the target return and downside risk remains acceptable."
        )
    elif final_decision == "INVEST WITH CONDITIONS":
        interpretation = (
            "Conditionally investable case. IRR is broadly supportive, "
            "but downside risk, valuation sensitivity, or structural assumptions require disciplined structuring."
        )
    else:
        interpretation = (
            "The current case does not meet the minimum investor-oriented return and downside requirements."
        )

    return {
        "FINAL_DECISION": final_decision,
        "Risk_Flag": risk_flag,
        "Interpretation": interpretation,
        "Hard_Gates_Passed": hard_gate_pass,
        "Return_Score": return_score,
        "Risk_Score": risk_score,
        "Total_Score": total_score,
    }


def build_scenario_table(base_config, base_df):
    scenario_inputs = [
        {
            "Scenario": "Upside",
            "Investor_Target_Return": safe_clip(base_config["investor_target_return"] + upside_rate_delta, low=0.01),
            "Exit_Multiple": safe_clip(base_config["exit_multiple"] + upside_exit_delta, low=1.0),
            "Cashflow_Volatility": safe_clip(base_config["sigma_cf"] + upside_vol_delta, low=0.01),
            "tail_event_prob": max(0.0, tail_event_prob * 0.5),
            "tail_event_severity": max(0.0, tail_event_severity * 0.75),
            "super_tail_prob": max(0.0, super_tail_prob * 0.5),
            "super_tail_severity": max(0.0, super_tail_severity * 0.75),
        },
        {
            "Scenario": "Base",
            "Investor_Target_Return": base_config["investor_target_return"],
            "Exit_Multiple": base_config["exit_multiple"],
            "Cashflow_Volatility": base_config["sigma_cf"],
            "tail_event_prob": tail_event_prob,
            "tail_event_severity": tail_event_severity,
            "super_tail_prob": super_tail_prob,
            "super_tail_severity": super_tail_severity,
        },
        {
            "Scenario": "High Rate",
            "Investor_Target_Return": safe_clip(base_config["investor_target_return"] + high_rate_delta, low=0.01),
            "Exit_Multiple": safe_clip(base_config["exit_multiple"] + high_rate_exit_delta, low=1.0),
            "Cashflow_Volatility": safe_clip(base_config["sigma_cf"] + high_rate_vol_delta, low=0.01),
            "tail_event_prob": tail_event_prob,
            "tail_event_severity": tail_event_severity,
            "super_tail_prob": super_tail_prob,
            "super_tail_severity": super_tail_severity,
        },
        {
            "Scenario": "Stress",
            "Investor_Target_Return": safe_clip(base_config["investor_target_return"] + stress_rate_delta, low=0.01),
            "Exit_Multiple": safe_clip(base_config["exit_multiple"] + stress_exit_delta, low=1.0),
            "Cashflow_Volatility": safe_clip(base_config["sigma_cf"] + stress_vol_delta, low=0.01),
            "tail_event_prob": min(1.0, tail_event_prob * 1.5),
            "tail_event_severity": min(1.0, tail_event_severity * 1.25),
            "super_tail_prob": min(1.0, super_tail_prob * 1.5),
            "super_tail_severity": min(1.0, super_tail_severity * 1.25),
        },
    ]

    rows = []

    for s in scenario_inputs:
        cfg = base_config.copy()
        cfg["investor_target_return"] = s["Investor_Target_Return"]
        cfg["exit_multiple"] = s["Exit_Multiple"]
        cfg["sigma_cf"] = s["Cashflow_Volatility"]
        cfg["tail_event_prob"] = s["tail_event_prob"]
        cfg["tail_event_severity"] = s["tail_event_severity"]
        cfg["super_tail_prob"] = s["super_tail_prob"]
        cfg["super_tail_severity"] = s["super_tail_severity"]
        cfg["scenario_name"] = s["Scenario"]

        res = run_sim(cfg, base_df)
        dec = make_decision(res["monte_carlo"], res["risk"], cfg["investor_target_return"])

        rows.append({
            "Scenario": s["Scenario"],
            "Investor Target Return": cfg["investor_target_return"],
            "Exit Multiple": cfg["exit_multiple"],
            "Volatility": cfg["sigma_cf"],
            "IRR Mean": res["monte_carlo"]["irr_mean"],
            "MOIC Mean": res["monte_carlo"]["moic_mean"],
            "NPV Mean": res["monte_carlo"]["npv_mean"],
            "Prob(NPV<0)": res["risk"]["prob_npv_negative"],
            "NPV CVaR (5%)": res["risk"]["npv_cvar_5"],
            "Decision": dec["FINAL_DECISION"],
        })

    return pd.DataFrame(rows).round(4)


def build_ic_summary(decision, mc, risk, gross_target_return):
    summary = f"""
Investor-oriented IC summary

Decision: {decision['FINAL_DECISION']}
Risk level: {decision['Risk_Flag']}

IRR mean: {mc['irr_mean']:.2%}
MOIC mean: {mc['moic_mean']:.2f}x
NPV mean: {mc['npv_mean']:.2f}

Probability of negative NPV: {risk['prob_npv_negative']:.2%}
NPV CVaR (5%): {risk['npv_cvar_5']:.2f}

Interpretation:
{decision['Interpretation']}

Illustrative gross target return required at fund level:
{gross_target_return:.2%}
""".strip()
    return summary.encode("utf-8")


def format_scenario_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    return out


# =========================
# Run logic
# =========================
if run_button:
    entry_ev = initial_cashflow * entry_multiple
    entry_equity = entry_ev - entry_debt

    base_config = {
        "initial_cashflow": initial_cashflow,
        "growth_rate": growth_rate,
        "contract_length": contract_length,
        "entry_multiple": entry_multiple,
        "exit_multiple": exit_multiple,
        "investor_target_return": investor_target_return,
        "entry_debt": entry_debt,
        "ltv_target": ltv_target,
        "entry_ev": entry_ev,
        "entry_equity": entry_equity,
        "implied_entry_ltv": entry_debt / entry_ev if entry_ev > 0 else np.nan,
        "operating_fee": operating_fee,
        "consortium_fee": consortium_fee,
        "n_simulations": int(n_simulations),
        "sigma_cf": cashflow_volatility,
        "investor_share": investor_share,
        "interest_rate": interest_rate,
        "tail_event_prob": tail_event_prob,
        "tail_event_severity": tail_event_severity,
        "super_tail_prob": super_tail_prob,
        "super_tail_severity": super_tail_severity,
    }

    results = run_sim(base_config, gt_clean)

    det = results["deterministic"]
    mc = results["monte_carlo"]
    risk = results["risk"]
    mc_raw = results["mc_raw"]
    overlay_stats = results["overlay_stats"]

    decision = make_decision(mc, risk, investor_target_return)

    gross_target_return = (investor_target_return + fund_mgmt_fee) / max(1e-6, (1 - fund_carry))

    scenario_df = build_scenario_table(base_config, gt_clean)

    tab1, tab2, tab3 = st.tabs(["Overview", "Scenarios", "Downloads"])

    with tab1:
        # ======================================
        # DEAL HEADER / IC SUMMARY
        # ======================================
        st.subheader("Deal Header")

        h1, h2, h3, h4 = st.columns(4)
        h1.metric(
            "IRR Mean",
            f"{mc['irr_mean']:.2%}",
            delta=f"{(mc['irr_mean'] - investor_target_return):.2%} vs target"
        )
        h2.metric(
            "MOIC Mean",
            f"{mc['moic_mean']:.2f}x",
            delta=f"{mc['moic_mean'] - 2.0:.2f}x vs 2.0x"
        )
        h3.metric("Expected NPV", f"{mc['npv_mean']:.2f}")
        h4.metric("Prob(NPV < 0)", f"{risk['prob_npv_negative']:.2%}")

        st.markdown("### Investment Decision Summary")

        dcol1, dcol2 = st.columns([1, 2])

        with dcol1:
            if decision["FINAL_DECISION"] == "INVEST":
                st.success(f"Decision: {decision['FINAL_DECISION']}")
            elif decision["FINAL_DECISION"] == "INVEST WITH CONDITIONS":
                st.warning(f"Decision: {decision['FINAL_DECISION']}")
            else:
                st.error(f"Decision: {decision['FINAL_DECISION']}")

            st.markdown(f"**Risk Flag:** {decision['Risk_Flag']}")
            st.markdown(f"**Hard Gates Passed:** {decision['Hard_Gates_Passed']}")

        with dcol2:
            st.markdown("**IC Interpretation**")
            st.write(decision["Interpretation"])

            st.markdown("**Deal Quality**")
            st.markdown(
                f"""
- IRR mean of **{mc['irr_mean']:.2%}** versus investor target return of **{investor_target_return:.2%}**
- MOIC mean of **{mc['moic_mean']:.2f}x**
- Expected NPV of **{mc['npv_mean']:.2f}**
- Downside probability of **{risk['prob_npv_negative']:.2%}**
"""
            )

        st.markdown("---")

        st.subheader("Risk & Distribution")

        dist_col1, dist_col2 = st.columns(2)

        with dist_col1:
            st.markdown("**IRR Distribution**")
            fig_irr, ax_irr = plt.subplots(figsize=(6, 3.5))
            ax_irr.hist(mc_raw["irr"], bins=30, alpha=0.85)
            ax_irr.axvline(investor_target_return, linestyle="--")
            ax_irr.axvline(np.mean(mc_raw["irr"]), linestyle=":")
            ax_irr.set_title("IRR Distribution")
            ax_irr.set_xlabel("IRR")
            ax_irr.set_ylabel("Frequency")
            ax_irr.grid(alpha=0.2)
            st.pyplot(fig_irr, width=450)

        with dist_col2:
            st.markdown("**NPV Distribution**")
            fig_npv, ax_npv = plt.subplots(figsize=(6, 3.5))
            ax_npv.hist(mc_raw["npv"], bins=30, alpha=0.85)
            ax_npv.axvline(0, linestyle="--")
            ax_npv.axvline(np.mean(mc_raw["npv"]), linestyle=":")
            ax_npv.set_title("NPV Distribution")
            ax_npv.set_xlabel("NPV")
            ax_npv.set_ylabel("Frequency")
            ax_npv.grid(alpha=0.2)

            annotation = (
                f"p(NPV<0) = {risk['prob_npv_negative']:.2%}\n"
                f"CVaR₅ = {risk['npv_cvar_5']:.2f}"
            )
            ax_npv.text(
                0.98, 0.95, annotation,
                transform=ax_npv.transAxes,
                ha="right", va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )

            st.pyplot(fig_npv, width=450)

        st.caption(
            "The dashed line marks the key threshold (target IRR or NPV = 0). The dotted line marks the simulated mean."
        )

        st.subheader("Downside & Fund-Level View")

        r1, r2, r3 = st.columns(3)
        r1.metric("Prob. Negative NPV", f"{risk['prob_npv_negative']:.2%}")
        r2.metric("NPV CVaR (5%)", f"{risk['npv_cvar_5']:.2f}")
        r3.metric("Investor Target Return", f"{investor_target_return:.2%}")

        g1, g2, g3 = st.columns(3)
        g1.metric("Investor Net Target", f"{investor_target_return:.2%}")
        g2.metric("Mgmt Fee Drag", f"{fund_mgmt_fee:.2%}")
        g3.metric("Illustrative Gross Target", f"{gross_target_return:.2%}")

        st.caption(
            "Illustrative bridge only. This is not a full fund waterfall model and should be interpreted as an indicative gross-to-net target return conversion."
        )

        det_df = pd.DataFrame([det]).round(4)
        mc_compact = pd.DataFrame([{
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

        st.subheader("Summary Tables")
        st.markdown("**Deterministic Summary**")
        st.dataframe(det_df, width="stretch")

        st.markdown("**Monte Carlo Highlights**")
        st.dataframe(mc_compact, width="stretch")

        st.subheader("Tail / Super-Tail Overlay")
        t1, t2 = st.columns(2)
        t1.metric("Tail Events Triggered", overlay_stats["tail_event_count"])
        t2.metric("Super-Tail Events Triggered", overlay_stats["super_tail_event_count"])

        st.caption(
            "This is a stylised downside-event overlay designed to distinguish regular tail risk from rarer super-tail outcomes."
        )

    with tab2:
        st.subheader("Scenario Comparison")

        scenario_display = format_scenario_table(scenario_df)
        st.dataframe(scenario_display, width="stretch")

        st.markdown("**Scenario Chart**")
        fig_scen, ax1 = plt.subplots(figsize=(8, 4))
        x = np.arange(len(scenario_df))
        ax1.bar(x, scenario_df["NPV Mean"])
        ax1.set_xticks(x)
        ax1.set_xticklabels(scenario_df["Scenario"])
        ax1.set_ylabel("NPV Mean")
        ax1.set_title("Scenario Comparison: NPV Mean and Prob(NPV<0)")
        ax1.grid(alpha=0.2)

        ax2 = ax1.twinx()
        ax2.plot(x, scenario_df["Prob(NPV<0)"], marker="o")
        ax2.set_ylabel("Prob(NPV<0)")

        st.pyplot(fig_scen, width="stretch")

        st.caption(
            "The four scenarios are user-configurable through the sidebar and are intended to test robustness across base, upside, high-rate, and stress conditions."
        )

    with tab3:
        st.subheader("Download Outputs")

        decision_df = pd.DataFrame([decision]).round(4)
        risk_df = pd.DataFrame([risk]).round(4)

        mc_compact = pd.DataFrame([{
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

        dl1, dl2, dl3, dl4 = st.columns(4)

        with dl1:
            st.download_button(
                label="Decision CSV",
                data=df_to_csv_bytes(decision_df),
                file_name="decision_summary.csv",
                mime="text/csv",
            )

        with dl2:
            st.download_button(
                label="Monte Carlo CSV",
                data=df_to_csv_bytes(mc_compact),
                file_name="monte_carlo_highlights.csv",
                mime="text/csv",
            )

        with dl3:
            st.download_button(
                label="Risk CSV",
                data=df_to_csv_bytes(risk_df),
                file_name="risk_metrics.csv",
                mime="text/csv",
            )

        with dl4:
            st.download_button(
                label="IC Summary TXT",
                data=build_ic_summary(decision, mc, risk, gross_target_return),
                file_name="ic_summary.txt",
                mime="text/plain",
            )

        st.subheader("Scenario CSV")
        st.download_button(
            label="Download Scenario Comparison CSV",
            data=df_to_csv_bytes(scenario_df),
            file_name="scenario_comparison.csv",
            mime="text/csv",
        )