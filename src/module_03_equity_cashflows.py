import numpy as np


def build_equity_cf(
    gross_cf,
    interest,
    amortisation,
    operating_fee_rate: float = 0.05,
    consortium_fee_rate: float = 0.0008,
    pg_share: float = 1.0,
    recap_proceeds=None,
):
    """
    Equity cashflows AFTER debt service on consortium level, then scaled by PG share.

    Equity_CF (consortium) =
        Gross_CF
      - OperatingFee
      - ConsortiumFee
      - Interest
      - Amortisation
      + Recap_Proceeds

    PG_Equity_CF = Equity_CF * PG_Share

    Returns dict with components and pg_equity_cf.
    """
    gross_cf = np.asarray(gross_cf, dtype=float)
    interest = np.asarray(interest, dtype=float)
    amortisation = np.asarray(amortisation, dtype=float)

    if recap_proceeds is None:
        recap_proceeds = np.zeros_like(gross_cf)
    recap_proceeds = np.asarray(recap_proceeds, dtype=float)

    operating_fee = gross_cf * float(operating_fee_rate)
    consortium_fee = gross_cf * float(consortium_fee_rate)

    equity_cf = gross_cf - operating_fee - consortium_fee - interest - amortisation + recap_proceeds
    pg_equity_cf = equity_cf * float(pg_share)

    return {
        "operating_fee": operating_fee,
        "consortium_fee": consortium_fee,
        "equity_cf": equity_cf,
        "pg_equity_cf": pg_equity_cf,
    }