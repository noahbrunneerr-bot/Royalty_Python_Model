import numpy as np


def _to_numpy_1d(values):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Input array must not be empty.")
    return arr


def moic(cashflows):
    """
    MOIC = total positive CF / abs(total negative CF)
    """
    cf = _to_numpy_1d(cashflows)
    pos = cf[cf > 0].sum()
    neg = -cf[cf < 0].sum()
    if neg == 0:
        return np.nan
    return float(pos / neg)


def irr_annual(cashflows, tol: float = 1e-10, max_iter: int = 200):
    """
    Robust annual IRR via bisection on NPV.
    Assumes equally spaced annual cashflows (t=0..N).
    Returns decimal (0.12 = 12%).
    """
    cf = _to_numpy_1d(cashflows)

    def npv(r):
        if r <= -0.999999999:
            return np.inf
        disc = (1.0 + r) ** np.arange(cf.size)
        return (cf / disc).sum()

    lo, hi = -0.9999, 10.0
    f_lo, f_hi = npv(lo), npv(hi)

    expand = 0
    while np.sign(f_lo) == np.sign(f_hi) and expand < 50:
        hi *= 2.0
        f_hi = npv(hi)
        expand += 1

    if np.sign(f_lo) == np.sign(f_hi):
        return np.nan

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = npv(mid)
        if abs(f_mid) < tol:
            return float(mid)
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid

    return float(0.5 * (lo + hi))


def payback_period(cashflows):
    """
    Returns the first period in which cumulative positive cashflows
    recover the initial investment.

    Assumes cashflows[0] is the initial negative investment.
    Returns:
        int | None
    """
    cf = _to_numpy_1d(cashflows)

    if cf.size == 0:
        return None

    initial_outflow = -cf[0] if cf[0] < 0 else 0.0
    if initial_outflow <= 0:
        return None

    cumulative_inflows = np.cumsum(cf[1:])

    for i, val in enumerate(cumulative_inflows, start=1):
        if val >= initial_outflow:
            return i

    return None


def total_positive_cf(cashflows):
    """
    Sum of all positive cashflows.
    """
    cf = _to_numpy_1d(cashflows)
    return float(cf[cf > 0].sum())


def total_negative_cf(cashflows):
    """
    Absolute sum of all negative cashflows.
    """
    cf = _to_numpy_1d(cashflows)
    return float(-cf[cf < 0].sum())


def discounted_cashflow_series(cashflows, discount_rate: float):
    """
    Discount each annual cash flow to present value.
    t=0 cash flow is not discounted.
    """
    cf = _to_numpy_1d(cashflows)

    if discount_rate <= -1.0:
        raise ValueError("discount_rate must be greater than -1.0")

    discount_factors = np.array(
        [1.0 / ((1.0 + discount_rate) ** t) for t in range(len(cf))],
        dtype=float,
    )
    return cf * discount_factors


def npv_annual(cashflows, discount_rate: float):
    """
    Net present value of annual cash flows using a constant annual discount rate.
    """
    dcf = discounted_cashflow_series(cashflows, discount_rate)
    return float(dcf.sum())


def profitability_index(cashflows, discount_rate: float):
    """
    Profitability Index = PV of future positive cash inflows / initial investment.
    Assumes initial investment occurs at t=0 and is negative.
    """
    cf = _to_numpy_1d(cashflows)

    if cf[0] >= 0:
        return np.nan

    dcf = discounted_cashflow_series(cf, discount_rate)
    pv_inflows = dcf[1:][dcf[1:] > 0].sum()
    initial_outflow = -dcf[0]

    if initial_outflow <= 0:
        return np.nan

    return float(pv_inflows / initial_outflow)


def deal_metrics(cashflows, discount_rate: float | None = None):
    """
    Standardized deal metrics block.

    Parameters
    ----------
    cashflows : array-like
        Full deal cashflow series including initial investment at t=0.

    Returns
    -------
    dict
        Dictionary with key valuation metrics.
    """
    cf = _to_numpy_1d(cashflows)

    out = {
        "investment": total_negative_cf(cf),
        "total_positive_cf": total_positive_cf(cf),
        "net_cf": float(cf.sum()),
        "moic": moic(cf),
        "irr": irr_annual(cf),
        "payback_period": payback_period(cf),
    }

    if discount_rate is not None:
        out["npv"] = npv_annual(cf, discount_rate)
        out["profitability_index"] = profitability_index(cf, discount_rate)

    return out


def probability_of_loss(values, threshold=0.0):
    """
    Probability that values fall below a threshold.
    """
    x = _to_numpy_1d(values)
    return float(np.mean(x < threshold))


def var_percentile(values, alpha=0.05):
    """
    Lower-tail percentile (VaR-style cutoff).
    Example: alpha=0.05 gives the 5th percentile.
    """
    x = _to_numpy_1d(values)
    return float(np.nanpercentile(x, alpha * 100))


def cvar_percentile(values, alpha=0.05):
    """
    Conditional Value at Risk / Expected Shortfall:
    average of observations below the alpha-percentile threshold.
    """
    x = _to_numpy_1d(values)
    cutoff = np.nanpercentile(x, alpha * 100)
    tail = x[x <= cutoff]
    if tail.size == 0:
        return np.nan
    return float(np.nanmean(tail))


def monte_carlo_risk_metrics(mc_out, alpha=0.05):
    """
    Risk summary for Monte Carlo output.
    Expects dict with keys:
      - irr
      - moic
      - net_cf
      - payback_period
    Optionally:
      - npv
    """
    irr = _to_numpy_1d(mc_out["irr"])
    moic_arr = _to_numpy_1d(mc_out["moic"])
    net_cf = _to_numpy_1d(mc_out["net_cf"])

    payback_raw = np.asarray(mc_out["payback_period"], dtype=float).reshape(-1)
    if payback_raw.size == 0:
        payback = np.array([np.nan], dtype=float)
    else:
        payback = payback_raw

    out = {
        "prob_irr_negative": probability_of_loss(irr, threshold=0.0),
        "prob_moic_below_1x": probability_of_loss(moic_arr, threshold=1.0),
        "prob_net_cf_negative": probability_of_loss(net_cf, threshold=0.0),
        "prob_no_payback": float(np.mean(np.isnan(payback))),
        "irr_var_5": var_percentile(irr, alpha=alpha),
        "irr_cvar_5": cvar_percentile(irr, alpha=alpha),
        "net_cf_var_5": var_percentile(net_cf, alpha=alpha),
        "net_cf_cvar_5": cvar_percentile(net_cf, alpha=alpha),
    }

    if "npv" in mc_out:
        npv_arr = _to_numpy_1d(mc_out["npv"])
        out.update({
            "prob_npv_negative": probability_of_loss(npv_arr, threshold=0.0),
            "npv_var_5": var_percentile(npv_arr, alpha=alpha),
            "npv_cvar_5": cvar_percentile(npv_arr, alpha=alpha),
        })

    return out