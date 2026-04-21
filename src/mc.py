import numpy as np
import pandas as pd

from src.module_01_cashflow_engine import build_royalty_cashflows
from src.module_02_debt_engine import build_debt_schedule
from src.module_03_equity_cashflows import build_equity_cf
from src.metrics import irr_annual, moic, deal_metrics, npv_annual


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns found: {candidates}")


def run_one_path(
    base_df: pd.DataFrame,
    exit_multiple: float,
    *,
    # column mapping (auto-pick defaults)
    col_gross_cf_candidates=None,
    col_nav_candidates=None,
    col_base_multiple_candidates=None,
    col_debt_end_candidates=None,
    col_equity_ticket_candidates=None,
    col_pg_share_candidates=None,
    # parameters
    operating_fee_rate: float = 0.05,
    consortium_fee_rate: float = 0.0008,
    interest_rate: float = 0.06,
    amort_rate: float = 0.0,
    cash_sweep_rate: float = 0.0,
    recap_year: int | None = None,
    recap_amount: float = 0.0,
    discount_rate: float = 0.10,
):
    """
    Single-path valuation using:
      - gross consortium CF (before fees/debt)
      - debt schedule on gross CF
      - equity CF after fees + debt service
      - terminal value from NAV * exit_multiple (scaled vs base multiple) minus debt_end

    Returns dict:
      exit_multiple, irr, moic, npv, cashflows, terminal_value
    """
    df = base_df.copy()

    if col_gross_cf_candidates is None:
        col_gross_cf_candidates = ["Net_CF_to_Consortium", "Net_CF", "NetCF", "Gross_CF"]
    if col_nav_candidates is None:
        col_nav_candidates = ["NAV"]
    if col_base_multiple_candidates is None:
        col_base_multiple_candidates = ["NAV_Multiple", "Exit_Multiple", "Base_Exit_Multiple"]
    if col_debt_end_candidates is None:
        col_debt_end_candidates = ["Debt_End", "Debt_EOP", "Debt", "Debt_Balance_End"]
    if col_equity_ticket_candidates is None:
        col_equity_ticket_candidates = ["Equity_Ticket", "EquityTicket", "Equity_Investment"]
    if col_pg_share_candidates is None:
        col_pg_share_candidates = ["PG_Share", "PGShare", "Ownership", "PG_Ownership"]

    col_gross_cf = _pick_col(df, col_gross_cf_candidates)
    col_nav = _pick_col(df, col_nav_candidates)
    col_base_mult = _pick_col(df, col_base_multiple_candidates)
    col_debt_end = _pick_col(df, col_debt_end_candidates)
    col_ticket = _pick_col(df, col_equity_ticket_candidates)

    pg_share = 1.0
    try:
        col_pg = _pick_col(df, col_pg_share_candidates)
        pg_share = float(df[col_pg].iloc[-1])
    except Exception:
        pg_share = 1.0

    gross_cf = np.asarray(df[col_gross_cf].to_numpy(), dtype=float)

    initial_debt = float(df[col_debt_end].iloc[0]) if col_debt_end in df.columns else 0.0
    debt = build_debt_schedule(
        gross_cf=gross_cf,
        initial_debt=initial_debt,
        interest_rate=interest_rate,
        amort_rate=amort_rate,
        cash_sweep_rate=cash_sweep_rate,
        recap_year=recap_year,
        recap_amount=recap_amount,
    )

    eq = build_equity_cf(
        gross_cf=gross_cf,
        interest=debt["interest"],
        amortisation=debt["amortisation"],
        operating_fee_rate=operating_fee_rate,
        consortium_fee_rate=consortium_fee_rate,
        pg_share=pg_share,
        recap_proceeds=debt["recap_proceeds"],
    )

    pg_equity_cf = eq["pg_equity_cf"].copy()

    nav_exit_base = float(df[col_nav].iloc[-1])
    base_mult = float(df[col_base_mult].iloc[-1])
    debt_end = float(df[col_debt_end].iloc[-1])

    nav_exit_sim = nav_exit_base * (float(exit_multiple) / base_mult)
    terminal_equity_value = max(0.0, nav_exit_sim - debt_end) * pg_share

    pg_equity_cf[-1] += terminal_equity_value

    ticket = float(df[col_ticket].iloc[0])
    cashflows = np.concatenate(([-ticket], pg_equity_cf))

    out_irr = irr_annual(cashflows)
    out_moic = moic(cashflows)
    out_npv = npv_annual(cashflows, discount_rate)

    return {
        "exit_multiple": float(exit_multiple),
        "irr": out_irr,
        "moic": out_moic,
        "npv": out_npv,
        "cashflows": cashflows,
        "terminal_value": terminal_equity_value,
    }


def simulate_base_series_paths(
    base_series,
    n_sim: int,
    sigma: float = 0.15,
    random_state: int | None = None,
):
    """
    Simulate stochastic base-series paths around a deterministic base path.
    """
    rng = np.random.default_rng(random_state)
    base = np.asarray(base_series, dtype=float)

    n_periods = len(base)
    shocks = rng.normal(loc=0.0, scale=sigma, size=(n_sim, n_periods))

    simulated = base * np.exp(shocks - 0.5 * sigma**2)

    return simulated


def run_pg3_monte_carlo(
    base_df: pd.DataFrame,
    exit_multiple_mean: float,
    exit_multiple_sigma: float,
    n_sim: int = 1000,
    random_state: int | None = 42,
    discount_rate: float = 0.10,
    sigma_cf: float = 0.15,
    **kwargs,
):
    """
    Monte Carlo wrapper around run_one_path(...) for the PG3 empirical case.

    Improvements:
    - stochastic exit multiple
    - stochastic cashflow path via lognormal shocks
    - stochastic NAV path via same shocks (keeps CF / terminal value linked)
    """
    rng = np.random.default_rng(random_state)

    # ----------------------------------------
    # 1. Draw exit multiples
    # ----------------------------------------
    exit_multiples = rng.normal(
        loc=exit_multiple_mean,
        scale=exit_multiple_sigma,
        size=n_sim,
    )
    exit_multiples = np.maximum(exit_multiples, 0.1)

    # ----------------------------------------
    # 2. Identify relevant columns
    # ----------------------------------------
    cf_col = _pick_col(base_df, ["Net_CF_to_Consortium", "Net_CF", "NetCF", "Gross_CF"])
    nav_col = _pick_col(base_df, ["NAV"])

    base_cf = np.asarray(base_df[cf_col].to_numpy(), dtype=float)
    base_nav = np.asarray(base_df[nav_col].to_numpy(), dtype=float)

    n_periods = len(base_cf)

    # ----------------------------------------
    # 3. Simulate CF shocks
    # ----------------------------------------
    shocks = rng.normal(loc=0.0, scale=sigma_cf, size=(n_sim, n_periods))
    shock_factors = np.exp(shocks - 0.5 * sigma_cf**2)

    irr_list = []
    moic_list = []
    npv_list = []
    net_cf_list = []
    cashflow_matrix = []
    terminal_values = []

    # ----------------------------------------
    # 4. Run simulation paths
    # ----------------------------------------
    for i, mult in enumerate(exit_multiples):
        df_path = base_df.copy()

        # cashflows shocked pathwise
        df_path[cf_col] = base_cf * shock_factors[i]

        # NAV shocked coherently with same path
        df_path[nav_col] = base_nav * shock_factors[i]

        out = run_one_path(
            base_df=df_path,
            exit_multiple=float(mult),
            discount_rate=discount_rate,
            **kwargs,
        )

        cf = np.asarray(out["cashflows"], dtype=float)

        irr_list.append(out["irr"])
        moic_list.append(out["moic"])
        npv_list.append(out["npv"])
        net_cf_list.append(float(cf.sum()))
        cashflow_matrix.append(cf)
        terminal_values.append(out["terminal_value"])

    return {
        "exit_multiple": np.asarray(exit_multiples, dtype=float),
        "irr": np.asarray(irr_list, dtype=float),
        "moic": np.asarray(moic_list, dtype=float),
        "npv": np.asarray(npv_list, dtype=float),
        "net_cf": np.asarray(net_cf_list, dtype=float),
        "cashflow_matrix": np.asarray(cashflow_matrix, dtype=float),
        "terminal_value": np.asarray(terminal_values, dtype=float),
        "discount_rate": float(discount_rate),
        "sigma_cf": float(sigma_cf),
    }


def summarize_royalty_mc(mc_out):
    """
    Summary statistics for royalty Monte Carlo output.
    """
    irr = mc_out["irr"]
    moic_arr = mc_out["moic"]
    net_cf = mc_out["net_cf"]

    out = {
        "irr_mean": float(np.nanmean(irr)),
        "irr_p10": float(np.nanpercentile(irr, 10)),
        "irr_p50": float(np.nanpercentile(irr, 50)),
        "irr_p90": float(np.nanpercentile(irr, 90)),
        "moic_mean": float(np.nanmean(moic_arr)),
        "moic_p10": float(np.nanpercentile(moic_arr, 10)),
        "moic_p50": float(np.nanpercentile(moic_arr, 50)),
        "moic_p90": float(np.nanpercentile(moic_arr, 90)),
        "net_cf_mean": float(np.nanmean(net_cf)),
        "net_cf_p10": float(np.nanpercentile(net_cf, 10)),
        "net_cf_p50": float(np.nanpercentile(net_cf, 50)),
        "net_cf_p90": float(np.nanpercentile(net_cf, 90)),
    }

    if "npv" in mc_out:
        npv_arr = mc_out["npv"]
        out.update({
            "npv_mean": float(np.nanmean(npv_arr)),
            "npv_p10": float(np.nanpercentile(npv_arr, 10)),
            "npv_p50": float(np.nanpercentile(npv_arr, 50)),
            "npv_p90": float(np.nanpercentile(npv_arr, 90)),
        })

    return out


def run_royalty_sensitivity(
    contract,
    base_series,
    initial_investments,
    royalty_rates,
    sigmas,
    n_sim: int = 1000,
    random_state: int | None = 42,
    risk_alpha: float = 0.05,
    discount_rate: float = 0.10,
):
    """
    Sensitivity runner for a generic royalty deal.
    """
    from src.metrics import monte_carlo_risk_metrics

    rows = []

    for investment in initial_investments:
        for rate in royalty_rates:
            for sigma in sigmas:
                scenario_contract = contract.__class__(
                    contract_name=contract.contract_name,
                    base_type=contract.base_type,
                    royalty_rate=rate,
                    start_period=contract.start_period,
                    end_period=contract.end_period,
                    payment_frequency=contract.payment_frequency,
                    payment_lag_periods=contract.payment_lag_periods,
                    cap_rule=contract.cap_rule,
                    floor_rule=contract.floor_rule,
                    termination_rule=contract.termination_rule,
                    step_up_rules=contract.step_up_rules,
                    milestones=contract.milestones,
                    catch_up_rule=contract.catch_up_rule,
                    uses_pg3_waterfall=contract.uses_pg3_waterfall,
                    uses_net_cf_to_consortium=contract.uses_net_cf_to_consortium,
                    debt_linked_distribution=contract.debt_linked_distribution,
                    exit_value_linked=contract.exit_value_linked,
                    metadata=contract.metadata,
                )

                mc_out = run_royalty_monte_carlo(
                    contract=scenario_contract,
                    base_series=base_series,
                    initial_investment=investment,
                    n_sim=n_sim,
                    sigma=sigma,
                    random_state=random_state,
                    discount_rate=discount_rate,
                )

                summary = summarize_royalty_mc(mc_out)
                risk = monte_carlo_risk_metrics(mc_out, alpha=risk_alpha)

                rows.append({
                    "initial_investment": investment,
                    "royalty_rate": rate,
                    "sigma": sigma,
                    "discount_rate": discount_rate,
                    **summary,
                    **risk,
                })

    return pd.DataFrame(rows)


def run_royalty_cap_floor_sensitivity(
    contract,
    base_series,
    initial_investment: float,
    floor_values,
    cap_values,
    n_sim: int = 1000,
    sigma: float = 0.15,
    random_state: int | None = 42,
    risk_alpha: float = 0.05,
    discount_rate: float = 0.10,
):
    """
    Sensitivity runner for floor/cap structures on a royalty deal.
    """
    from src.contracts.royalty_contract import CapRule, FloorRule
    from src.metrics import monte_carlo_risk_metrics

    rows = []

    for floor_val in floor_values:
        for cap_val in cap_values:
            floor_rule = FloorRule(
                active=(floor_val is not None and floor_val > 0),
                floor_type="annual_minimum" if floor_val is not None and floor_val > 0 else "none",
                floor_value=float(floor_val) if floor_val is not None and floor_val > 0 else None,
            )

            cap_rule = CapRule(
                active=(cap_val is not None),
                cap_type="total_multiple" if cap_val is not None else "none",
                cap_value=float(cap_val) if cap_val is not None else None,
            )

            scenario_contract = contract.__class__(
                contract_name=contract.contract_name,
                base_type=contract.base_type,
                royalty_rate=contract.royalty_rate,
                start_period=contract.start_period,
                end_period=contract.end_period,
                payment_frequency=contract.payment_frequency,
                payment_lag_periods=contract.payment_lag_periods,
                cap_rule=cap_rule,
                floor_rule=floor_rule,
                termination_rule=contract.termination_rule,
                step_up_rules=contract.step_up_rules,
                milestones=contract.milestones,
                catch_up_rule=contract.catch_up_rule,
                uses_pg3_waterfall=contract.uses_pg3_waterfall,
                uses_net_cf_to_consortium=contract.uses_net_cf_to_consortium,
                debt_linked_distribution=contract.debt_linked_distribution,
                exit_value_linked=contract.exit_value_linked,
                metadata=contract.metadata,
            )

            mc_out = run_royalty_monte_carlo(
                contract=scenario_contract,
                base_series=base_series,
                initial_investment=initial_investment,
                n_sim=n_sim,
                sigma=sigma,
                random_state=random_state,
                discount_rate=discount_rate,
            )

            summary = summarize_royalty_mc(mc_out)
            risk = monte_carlo_risk_metrics(mc_out, alpha=risk_alpha)

            rows.append({
                "floor_value": floor_val,
                "cap_multiple": cap_val,
                "sigma": sigma,
                "discount_rate": discount_rate,
                **summary,
                **risk,
            })

    return pd.DataFrame(rows)


def summarize_pg3_mc(mc_out):
    """
    Summary statistics for PG3 Monte Carlo output.
    """
    irr = mc_out["irr"]
    moic_arr = mc_out["moic"]
    net_cf = mc_out["net_cf"]
    term_val = mc_out["terminal_value"]

    out = {
        "irr_mean": float(np.nanmean(irr)),
        "irr_p10": float(np.nanpercentile(irr, 10)),
        "irr_p50": float(np.nanpercentile(irr, 50)),
        "irr_p90": float(np.nanpercentile(irr, 90)),
        "moic_mean": float(np.nanmean(moic_arr)),
        "moic_p10": float(np.nanpercentile(moic_arr, 10)),
        "moic_p50": float(np.nanpercentile(moic_arr, 50)),
        "moic_p90": float(np.nanpercentile(moic_arr, 90)),
        "net_cf_mean": float(np.nanmean(net_cf)),
        "net_cf_p10": float(np.nanpercentile(net_cf, 10)),
        "net_cf_p50": float(np.nanpercentile(net_cf, 50)),
        "net_cf_p90": float(np.nanpercentile(net_cf, 90)),
        "terminal_value_mean": float(np.nanmean(term_val)),
        "terminal_value_p10": float(np.nanpercentile(term_val, 10)),
        "terminal_value_p50": float(np.nanpercentile(term_val, 50)),
        "terminal_value_p90": float(np.nanpercentile(term_val, 90)),
    }

    if "npv" in mc_out:
        npv_arr = mc_out["npv"]
        out.update({
            "npv_mean": float(np.nanmean(npv_arr)),
            "npv_p10": float(np.nanpercentile(npv_arr, 10)),
            "npv_p50": float(np.nanpercentile(npv_arr, 50)),
            "npv_p90": float(np.nanpercentile(npv_arr, 90)),
        })

    return out