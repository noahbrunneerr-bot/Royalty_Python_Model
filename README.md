# Royalty Valuation Tool

## Overview

This project contains a Python-based royalty valuation and decision-support tool developed in the context of a bachelor thesis. It combines deterministic valuation, Monte Carlo simulation, scenario analysis, downside risk assessment, and an investor-oriented decision layer.

The current implementation is calibrated to a real-case structure and demonstrates how royalty investments can be assessed from both a valuation and investment committee perspective.

For presentation consistency, the technical label **Discount Rate** is used in the front-end and sensitivity tables.
From an economic perspective, this parameter represents the return requirement applied to the valuation and can therefore be interpreted as an investor hurdle-rate proxy.

## Main Components

The project consists of two main layers:

1. **Notebook-based model layer**
   - deterministic valuation
   - calibration against reference values
   - Monte Carlo simulation
   - scenario analysis
   - driver analysis
   - contract intelligence
   - decision framework

2. **Streamlit front-end**
   - valuation input interface with investor-oriented interpretation
   - key metrics and risk metrics
   - NPV distribution chart
   - scenario comparison
   - investment decision view
   - downloadable output files

## Folder Structure

- `app/`  
  Streamlit application and local launch files

- `src/`  
  Core model logic and supporting functions

- `Data/processed/`  
  Processed input data used by the model

- `notebooks/`  
  Development and final notebook versions

- `outputs/`  
  Exported output files from notebook runs

## Main Workflow

### Option 1: Notebook workflow

Use the notebook if you want to review the full modelling logic and all analytical blocks.

Recommended final notebook:
- `notebooks/04_final_pg3_tool.ipynb`

Suggested workflow:
1. Open the final notebook
2. Restart kernel
3. Run all cells from top to bottom
4. Review the main output sections:
   - Calibration Check
   - Monte Carlo Simulation
   - Scenario & Macro Analysis
   - Driver Analysis
   - Contract Intelligence
   - Investment Decision Framework
   - Management Summary
5. Review exported output files in `/outputs`

### Option 2: Streamlit tool

Use the Streamlit application if you want an investor-facing interface for practical case testing.
In the Streamlit front-end, the model uses the term **Discount Rate** for technical consistency. 
Economically, this discount rate can be interpreted as a practical proxy for the investor’s required return / hurdle rate.

Open a terminal and navigate to the app folder:

```bash
cd C:\Users\Noah\PG3_Royalty_Python_Model\app