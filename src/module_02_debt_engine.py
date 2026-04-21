import numpy as np

def build_debt_schedule(
    gross_cf,
    initial_debt,
    interest_rate=0.06,
    amort_rate=0.0,
    cash_sweep_rate=0.0,
    recap_year=None,
    recap_amount=0.0,
):
    """
    Compatible debt schedule for run_one_path()

    Parameters
    ----------
    gross_cf : array-like
    initial_debt : float
    interest_rate : float
    amort_rate : float
    cash_sweep_rate : float
    recap_year : int or None
    recap_amount : float
    """

    gross_cf = np.array(gross_cf, dtype=float)
    n = len(gross_cf)

    debt = np.zeros(n)
    interest = np.zeros(n)
    amort = np.zeros(n)
    recap_proceeds = np.zeros(n)

    debt[0] = initial_debt

    for t in range(n):

        # Interest
        interest[t] = debt[t] * interest_rate

        # Amortisation
        amort[t] = debt[t] * amort_rate

        # Cash sweep
        cash_sweep = gross_cf[t] * cash_sweep_rate

        # Recap event
        if recap_year is not None and t == recap_year:
            recap_proceeds[t] = recap_amount
            debt[t] += recap_amount

        # Update next year debt
        if t < n - 1:
            debt[t + 1] = max(
                0.0,
                debt[t] - amort[t] - cash_sweep
            )

    return {
        "debt_balance": debt,
        "interest": interest,
        "amortisation": amort,
        "recap_proceeds": recap_proceeds,
    }