# PG3 Final Royalty Valuation Tool

This notebook provides a Python-based valuation and risk assessment tool for the PG3 royalty case. It combines deterministic calibration against Excel ground truth with Monte Carlo simulation, macro scenario analysis, driver analysis and an investment decision framework.

## Purpose

The tool is designed to support investment evaluation for royalty-based transactions by integrating:
- deterministic base-case valuation
- stochastic risk analysis
- macro and discount-rate sensitivity
- driver-based scenario testing
- a structured investment recommendation

## Main Workflow

1. Open the notebook `03_final_pg3_tool.ipynb`
2. Restart kernel
3. Run all cells from top to bottom
4. Review the main output sections:
   - Calibration Check
   - Monte Carlo Simulation
   - Scenario & Macro Analysis
   - Driver Analysis
   - Investment Decision Framework
   - Management Summary
5. Review exported output files in `/outputs`

## User Inputs

The main user-defined assumptions are set in the `DEAL_INPUT` block, including:
- initial cashflow
- growth rate
- contract length
- entry / exit multiple
- entry debt / target LTV
- operating and consortium fee
- discount rate
- simulation count
- volatility

## Main Outputs

The tool produces the following key outputs:
- calibration summary against Excel ground truth
- deterministic deal-level valuation
- Monte Carlo return and risk metrics
- macro and discount-rate sensitivity results
- driver analysis
- investment recommendation
- management summary and exported CSV outputs

## Exported Files

The notebook exports the following files to `/outputs`:
- `ic_summary.csv`
- `decision_summary.csv`
- `driver_analysis.csv`
- `driver_impact_summary.csv`
- `scenario_macro_summary.csv`
- `discount_rate_sweep.csv`
- `calibration_summary.csv`
- `management_summary.txt`

## Important Interpretation Note

This tool is designed as a decision-support system. It does not replace full legal, commercial or contractual due diligence. Investment conclusions should always be interpreted together with qualitative review and downside protection considerations.