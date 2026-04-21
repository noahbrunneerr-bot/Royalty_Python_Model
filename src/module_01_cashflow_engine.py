import numpy as np
from src.contracts.royalty_contract import RoyaltyContract


def build_operating_cf(revenue, ebitda_margin, capex_pct, tax_rate):
    """
    Build operating cash flow series from revenue assumptions.

    Parameters
    ----------
    revenue : array-like
        Revenue series by period.
    ebitda_margin : float
        EBITDA margin as decimal (e.g. 0.25 = 25%).
    capex_pct : float
        Capex as percentage of revenue.
    tax_rate : float
        Tax rate as decimal.

    Returns
    -------
    dict
        Dictionary with EBITDA, EBIT, taxes and operating CF series.
    """
    revenue = np.asarray(revenue, dtype=float)

    ebitda = revenue * ebitda_margin
    capex = revenue * capex_pct
    ebit = ebitda - capex
    taxes = np.maximum(ebit, 0) * tax_rate
    operating_cf = ebitda - capex - taxes

    return {
        "revenue": revenue,
        "ebitda": ebitda,
        "capex": capex,
        "ebit": ebit,
        "taxes": taxes,
        "operating_cf": operating_cf,
    }


def apply_payment_lag(cashflows, lag_periods: int):
    """
    Shift cashflows forward by lag_periods.
    Example: lag_periods=1 means period t cashflow is paid in t+1.

    Parameters
    ----------
    cashflows : array-like
        Period cashflows before payment lag.
    lag_periods : int
        Number of periods to shift forward.

    Returns
    -------
    np.ndarray
        Lagged cashflow array of same length.
    """
    cf = np.asarray(cashflows, dtype=float)

    if lag_periods < 0:
        raise ValueError("lag_periods must be >= 0.")

    if lag_periods == 0:
        return cf.copy()

    lagged = np.zeros_like(cf, dtype=float)
    if lag_periods < len(cf):
        lagged[lag_periods:] = cf[:-lag_periods]

    return lagged


def build_royalty_cashflows(
    contract: RoyaltyContract,
    base_series,
    initial_investment: float | None = None,
):
    """
    Generic royalty cashflow builder based on a RoyaltyContract.

    Parameters
    ----------
    contract : RoyaltyContract
        Contract object defining royalty logic.
    base_series : array-like
        Base amounts by period (e.g. revenue, gross profit, net cash).
        Length must match contract term length.
    initial_investment : float, optional
        Used only if cap_type='total_multiple' is active.

    Returns
    -------
    dict with arrays:
        base_series
        royalty_rate
        pre_floor_cf
        floor_top_up
        post_floor_cf
        milestone_cf
        pre_cap_cf
        final_cf_pre_lag
        final_cf
        cap_remaining
    """
    contract.validate()

    base_series = np.asarray(base_series, dtype=float)
    n_periods = contract.term_length

    if len(base_series) != n_periods:
        raise ValueError(
            f"Length of base_series ({len(base_series)}) must equal contract term_length ({n_periods})."
        )

    royalty_rates = np.array(
        [contract.get_applicable_rate(x) for x in base_series],
        dtype=float,
    )

    pre_floor_cf = base_series * royalty_rates

    floor_top_up = np.zeros(n_periods, dtype=float)
    post_floor_cf = pre_floor_cf.copy()

    if contract.floor_rule.active:
        if contract.floor_rule.floor_type == "annual_minimum":
            floor_value = float(contract.floor_rule.floor_value)
            floor_top_up = np.maximum(0.0, floor_value - pre_floor_cf)
            post_floor_cf = pre_floor_cf + floor_top_up

        elif contract.floor_rule.floor_type == "total_minimum":
            total_minimum = float(contract.floor_rule.floor_value)
            shortfall = max(0.0, total_minimum - float(pre_floor_cf.sum()))
            floor_top_up[-1] = shortfall
            post_floor_cf = pre_floor_cf + floor_top_up

    milestone_cf = np.array(
        [contract.get_milestone_amount(period=i) for i in range(n_periods)],
        dtype=float,
    )

    pre_cap_cf = post_floor_cf + milestone_cf

    final_cf_pre_lag = pre_cap_cf.copy()
    cap_remaining = np.full(n_periods, np.nan)

    if contract.cap_rule.active:
        if contract.cap_rule.cap_type == "absolute_amount":
            cap_total = float(contract.cap_rule.cap_value)

        elif contract.cap_rule.cap_type == "total_multiple":
            if initial_investment is None:
                raise ValueError(
                    "initial_investment must be provided when cap_type='total_multiple'."
                )
            cap_total = float(contract.cap_rule.cap_value) * float(initial_investment)

        else:
            raise ValueError(f"Unsupported cap_type: {contract.cap_rule.cap_type}")

        cumulative = 0.0
        for i in range(n_periods):
            allowed = max(0.0, cap_total - cumulative)
            final_cf_pre_lag[i] = min(pre_cap_cf[i], allowed)
            cumulative += final_cf_pre_lag[i]
            cap_remaining[i] = max(0.0, cap_total - cumulative)

    final_cf = apply_payment_lag(final_cf_pre_lag, contract.payment_lag_periods)

    return {
        "base_series": base_series,
        "royalty_rate": royalty_rates,
        "pre_floor_cf": pre_floor_cf,
        "floor_top_up": floor_top_up,
        "post_floor_cf": post_floor_cf,
        "milestone_cf": milestone_cf,
        "pre_cap_cf": pre_cap_cf,
        "final_cf_pre_lag": final_cf_pre_lag,
        "final_cf": final_cf,
        "cap_remaining": cap_remaining,
    }