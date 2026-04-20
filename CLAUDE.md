# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
streamlit run app.py
```

No build step. No tests. No linter configured.

## Architecture

This is a **single-file Streamlit app** (`app.py`). All logic lives in one file in this order:

1. **Page config + CSS injection** — custom dark theme (navy sidebar, gold KPI cards, Inter font). All styling is injected via `st.markdown()` at the top.
2. **`MASTER_INTEL` dict** — the central data registry. Maps stock name → `{Rate, PB, Ticker, Sector}`. `Rate` is DPS (dividends per share in SGD), `PB` is price-to-book ratio, `Ticker` is the Yahoo Finance `.SI` symbol.
3. **Helper functions** — `clean_val`, `guess_sector`, `resolve_intel`, `extract_pdf`, `color_returns`, `kpi_card`, `health_gauge`, `project_dividends`.
4. **Cached data-fetch / compute functions** — all use `@st.cache_data(ttl=3600)`:
   - `fetch_prices_batch(tickers_str, period)` — batch-downloads closing prices for multiple tickers; returns a DataFrame with tickers as columns. Falls back to per-ticker download if the batch response is malformed.
   - `compute_risk_metrics(tickers_str, names_str)` — computes annualised volatility, return, Sharpe ratio (3.7% risk-free), beta vs STI, daily VaR 95%, and max drawdown per holding.
   - `get_dividend_data(tickers_str, names_str)` — fetches `yf.Ticker.dividends` for each holding; returns `{name: pd.Series}`.
   - `compute_efficient_frontier(tickers_str, names_str, weights_str)` — runs 3,000 Monte Carlo portfolios on 3-year price history; returns a dict with all simulation results plus `curr`, `max_sharpe`, and `min_vol` sub-dicts.
   - `scan_portfolio_returns(_data_hash, tickers_and_names)` — 5-year total return per holding (used by Strategic Advisory tab).
   - `load_benchmark(ticker, years)` — single-ticker vs STI price history (used by Market Benchmark tab).
5. **`project_dividends(div_series, qty, n_months)`** — non-cached helper. Detects payout frequency from historical intervals (median of gaps), projects next `n_months` of income.
6. **Sidebar** — branding header + PDF uploader + sector filter multiselect + maintenance guide expander.
7. **Seven tabs** — Asset Discovery, Market Benchmark, Verification Hub, Strategic Advisory, Risk Analytics, Dividend Calendar, Portfolio Optimisation.
8. **Landing page** — shown when no PDF is uploaded; feature-card HTML rendered via `st.markdown(unsafe_allow_html=True)`.

## Key Conventions

**Adding a new stock:** Add an entry to `MASTER_INTEL` at the top of `app.py`. Without this, a stock parsed from the PDF will still appear in Asset Discovery and Verification Hub but will be excluded from Market Benchmark, Risk Analytics, Dividend Calendar, Portfolio Optimisation, and Strategic Advisory tabs (because `Ticker` will be `None`).

**PDF parsing:** `extract_pdf()` uses `pdfplumber` to walk every table on every page. It calls `resolve_intel()` which first does an exact-match lookup in `MASTER_INTEL`, then falls back to `ALIAS_MAP` for common name variations, then falls back to `guess_sector()`. Duplicates are skipped via a `seen` set.

**Data flow:** The parsed `df_raw` is filtered by the sidebar sector multiselect → `df`. All downstream computed columns (`Annual Dividend (SGD)`, `Dividend Yield (%)`, `%Portfolio`, `%Income`, `Alloc_Label`, `Inc_Label`) are added to `df` after filtering. All tabs read from `df`, and tabs that need market data filter further to `df[df['Ticker'].notnull()]`.

**Market data:** `yfinance` is used for all live price/history fetching. The STI benchmark ticker is `^STI`. All `.SI` tickers are Singapore Exchange symbols. `auto_adjust=True` is used throughout for split/dividend-adjusted prices.

**Batch price downloads:** `fetch_prices_batch` handles both single and multi-ticker yfinance responses. When yfinance returns a MultiIndex (multi-ticker case), it extracts `raw['Close'][ticker]`; for a single ticker, it extracts `raw['Close']`. There is a per-ticker fallback loop for any tickers that fail in the batch call.

**Risk Analytics tab:** Auto-computes on every page load (cached). Displays VaR as both percentage and SGD dollar amount (VaR% × AUM). The portfolio-level VaR shown in the KPI is the sum of individual dollar VaRs — this is an upper bound that overstates risk because it ignores diversification benefit.

**Dividend Calendar tab:** `project_dividends` strips timezone from the dividend series index to avoid tz-aware/tz-naive comparison errors, then computes the median inter-payment interval to determine frequency. Projects from the last known payment date.

**Portfolio Optimisation tab:** The Efficient Frontier computation is placed behind a button (not auto-triggered) because it requires 3-year price history downloads for all holdings. Results are cached — the spinner only appears on first run per session. The current portfolio weights fed into the simulation are the AUM-proportional weights of `Ticker`-eligible holdings only (non-ticker holdings are excluded from the optimisation).

**Rebalance simulator:** Fetches the latest 5-day price for each buy target to derive a live yield (`DPS / live_price`), then computes projected income by applying that yield to the freed capital. Falls back to a 4% assumption if the fetch fails.
