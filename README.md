# Spinoff Index Arbitrage

Systematic event-driven strategy to exploit price dislocations around index inclusion and subsequent deletion of spin-off securities.

## Overview

This project investigates whether index mechanics and eligibility rules create predictable short-horizon return patterns that can be monetized through a market-neutral trading strategy.

The strategy focuses on:

* Inclusion-driven buying pressure
* Deletion-driven selling pressure
* Mean reversion after forced flows
* Beta-neutral portfolio construction

## Data Sources

* Corporate actions (spin-off announcements, ex-dates, filings)
* Index methodology and constituent changes
* Market and microstructure data (prices, volume, bid-ask spreads, volatility)
* News and analyst commentary

## Strategy

* **Entry:** Short the spin-off at the first trading day close when inclusion-related flow signals trigger.
* **Hedge:** Long broad equity index futures to target beta neutrality.
* **Exit:** Deletion confirmation, rebalance date, time-stop (21–60 trading days), or stop-loss.
* **Position sizing:** Based on a deletion probability model using eligibility and liquidity features.

## Project Timeline

1. Define universe
2. Source datasets
3. EDA and statistical testing
4. Build v1 signals
5. Refine signals and generate trades
6. Backtest strategy
7. Visualize results
8. Finalize MVP and presentation

## Environment

This repo uses **Python == 3.12.13**.

## Installation

### 1. Virtual Environment

```bash
conda env create -f environment.yaml
conda activate index_spinoff_arbitrage_env
```
