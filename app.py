import streamlit as st
import pandas as pd
import pdfplumber
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import re
import numpy as np
from datetime import datetime, timedelta

# --- 1. SYSTEM CONFIG ---
st.set_page_config(layout="wide", page_title="CLP Wealth Center", page_icon="🏦")

# --- 2. CUSTOM CSS THEME ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071428 0%, #0d2240 60%, #0a1a35 100%);
}
section[data-testid="stSidebar"] * { color: #dce8f5 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #f0b429 !important; }
section[data-testid="stSidebar"] hr { border-color: #1e3a6b !important; }
section[data-testid="stSidebar"] .stButton > button {
    background: transparent; border: 1px solid #f0b429;
    color: #f0b429 !important; border-radius: 8px; width: 100%;
    font-weight: 600; letter-spacing: 0.5px;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(240, 180, 41, 0.12);
}
section[data-testid="stSidebar"] label { color: #8fa8cc !important; font-size: 0.8rem !important; }

.kpi-card {
    background: linear-gradient(135deg, #0d2240 0%, #1a3a6b 100%);
    border: 1px solid #2a4a8b; border-radius: 14px;
    padding: 22px 20px 18px; text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.kpi-icon { font-size: 1.5rem; margin-bottom: 6px; }
.kpi-label { font-size: 0.7rem; color: #8fa8cc; text-transform: uppercase; letter-spacing: 1.4px; margin-bottom: 8px; }
.kpi-value { font-size: 1.85rem; font-weight: 700; color: #f0b429; line-height: 1.1; }

.tab-subtitle {
    color: #6b7f9e; font-size: 0.87rem;
    margin: -6px 0 20px 0; padding-bottom: 14px;
    border-bottom: 1px solid #eaecf0;
}
.conc-alert {
    background: rgba(240, 180, 41, 0.08); border-left: 4px solid #f0b429;
    border-radius: 0 8px 8px 0; padding: 12px 16px;
    margin: 0 0 20px 0; font-size: 0.88rem; color: #7a5c00;
}
.feat-grid { display: flex; gap: 14px; margin: 28px 0 10px 0; flex-wrap: wrap; }
.feat-card {
    flex: 1; min-width: 160px;
    background: #f4f8ff; border: 1px solid #d0e2f7;
    border-radius: 14px; padding: 22px 16px; text-align: center;
}
.feat-icon { font-size: 2rem; margin-bottom: 10px; }
.feat-title { font-weight: 600; font-size: 0.92rem; color: #0d2240; margin-bottom: 6px; }
.feat-desc { font-size: 0.8rem; color: #6b7f9e; line-height: 1.4; }
.hide-delta [data-testid="stMetricDelta"] svg { display: none; }

/* ── Tab navigation ─────────────────────────────────────────── */
div[data-baseweb="tab-list"] {
    background: linear-gradient(180deg, #eef3fb 0%, #e8eef8 100%);
    border-radius: 12px 12px 0 0;
    padding: 6px 6px 0 6px;
    gap: 2px;
}
button[data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px 8px 0 0 !important;
    color: #6b7f9e !important;
    font-size: 0.81rem !important;
    font-weight: 500 !important;
    padding: 8px 15px 10px 15px !important;
    white-space: nowrap;
    transition: background 0.15s ease, color 0.15s ease;
}
button[data-baseweb="tab"]:hover {
    background: rgba(240, 180, 41, 0.10) !important;
    color: #0d2240 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: white !important;
    color: #0d2240 !important;
    font-weight: 700 !important;
}
div[data-baseweb="tab-highlight"] {
    background-color: #f0b429 !important;
    height: 3px !important;
    border-radius: 3px 3px 0 0;
}
div[data-baseweb="tab-border"] {
    background: #c8d8ec !important;
    height: 1px !important;
}
</style>
""", unsafe_allow_html=True)

# --- 3. ASSET REGISTRY ---
# _DPS_AS_OF: the fiscal year from which all Rate values were sourced.
# UPDATE this string whenever you refresh any Rate value.
_DPS_AS_OF = "FY 2024"

# Per-holding caveats shown as warnings throughout the app.
# Add an entry here whenever a Rate requires user attention.
DPS_NOTES = {
    "DBS": (
        "Rate updated to S$2.22 (FY 2024 ordinary only). "
        "DBS also pays a recurring S$0.50/share special capital return — "
        "add it back to Rate if you wish to include it in income projections."
    ),
    "HPH TRUST USD": (
        "Distributions are paid in USD, not SGD. "
        "Rate 0.02 is used here as SGD — cross-check against the prevailing USD/SGD rate "
        "and adjust Rate to the SGD-equivalent for accurate income figures."
    ),
    "KEPPEL": (
        "Keppel Corp completed a major restructuring in 2023. "
        "Verify Rate against the latest post-restructuring annual report DPS."
    ),
    "MAPLETREE LOG TR": (
        "Mapletree Logistics has been reducing DPU in recent quarters due to rising costs. "
        "Verify Rate against the latest declared DPU before relying on income projections."
    ),
}

MASTER_INTEL = {
    "DBS": {"Rate": 3.24, "PB": 2.34, "Ticker": "D05.SI", "Sector": "Financial Services"},
    "UOB": {"Rate": 1.70, "PB": 1.18, "Ticker": "U11.SI", "Sector": "Financial Services"},
    "OCBC": {"Rate": 0.86, "PB": 1.10, "Ticker": "O39.SI", "Sector": "Financial Services"},
    "SINGTEL": {"Rate": 0.16, "PB": 1.32, "Ticker": "Z74.SI", "Sector": "Telecommunications"},
    "SUNTEC REIT": {"Rate": 0.08, "PB": 0.62, "Ticker": "T82U.SI", "Sector": "REITs & Business Trusts"},
    "ST ENGINEERING": {"Rate": 0.16, "PB": 4.50, "Ticker": "S63.SI", "Sector": "Industrials & Diversified"},
    "KEPPEL": {"Rate": 0.47, "PB": 1.15, "Ticker": "BN4.SI", "Sector": "Industrials & Diversified"},
    "MAPLETREE LOG TR": {"Rate": 0.08, "PB": 0.95, "Ticker": "M44U.SI", "Sector": "REITs & Business Trusts"},
    "VENTURE": {"Rate": 0.75, "PB": 1.22, "Ticker": "V03.SI", "Sector": "Technology"},
    "WILMAR INTL": {"Rate": 0.17, "PB": 0.78, "Ticker": "F34.SI", "Sector": "Consumer Goods"},
    "COMFORTDELGRO": {"Rate": 0.07, "PB": 1.05, "Ticker": "C52.SI", "Sector": "Industrials & Diversified"},
    "NETLINK NBN TR": {"Rate": 0.05, "PB": 1.35, "Ticker": "CJLU.SI", "Sector": "REITs & Business Trusts"},
    "KEP INFRA TR": {"Rate": 0.04, "PB": 2.10, "Ticker": "A7RU.SI", "Sector": "REITs & Business Trusts"},
    "SATS": {"Rate": 0.05, "PB": 1.85, "Ticker": "S58.SI", "Sector": "Industrials & Diversified"},
    "SEMBCORP IND": {"Rate": 0.15, "PB": 1.42, "Ticker": "U96.SI", "Sector": "Industrials & Diversified"},
    "SIA ENGINEERING": {"Rate": 0.08, "PB": 1.25, "Ticker": "S59.SI", "Sector": "Industrials & Diversified"},
    "CITYDEV": {"Rate": 0.12, "PB": 0.38, "Ticker": "C09.SI", "Sector": "Real Estate"},
    "CAPLAND ASCOTT T": {"Rate": 0.07, "PB": 0.88, "Ticker": "HMN.SI", "Sector": "REITs & Business Trusts"},
    "ASIAN PAY TV TR": {"Rate": 0.01, "PB": 0.45, "Ticker": "S7OU.SI", "Sector": "Telecommunications"},
    "CAPITA CHINA TR": {"Rate": 0.07, "PB": 0.62, "Ticker": "AU8U.SI", "Sector": "REITs & Business Trusts"},
    "CAPLAND INDIA T": {"Rate": 0.08, "PB": 1.12, "Ticker": "CY6U.SI", "Sector": "REITs & Business Trusts"},
    "CDL HTRUST": {"Rate": 0.06, "PB": 0.75, "Ticker": "J85.SI", "Sector": "REITs & Business Trusts"},
    "FIRST REIT": {"Rate": 0.02, "PB": 0.55, "Ticker": "AW9U.SI", "Sector": "REITs & Business Trusts"},
    "KEPPEL REIT": {"Rate": 0.06, "PB": 0.68, "Ticker": "K71U.SI", "Sector": "REITs & Business Trusts"},
    "KIMLY": {"Rate": 0.02, "PB": 1.45, "Ticker": "1D0.SI", "Sector": "Consumer Goods"},
    "LIPPO MALLS TR": {"Rate": 0.00, "PB": 0.12, "Ticker": "D5IU.SI", "Sector": "REITs & Business Trusts"},
    "OLAM GROUP": {"Rate": 0.07, "PB": 0.58, "Ticker": "VC2.SI", "Sector": "Consumer Goods"},
    "QUEREIT": {"Rate": 0.03, "PB": 0.42, "Ticker": "BJW.SI", "Sector": "REITs & Business Trusts"},
    "SEATRIUM LTD": {"Rate": 0.02, "PB": 1.10, "Ticker": "S51.SI", "Sector": "Industrials & Diversified"},
    "SINGPOST": {"Rate": 0.01, "PB": 0.65, "Ticker": "S08.SI", "Sector": "Industrials & Diversified"},
    "STARHILLGBL REIT": {"Rate": 0.04, "PB": 0.72, "Ticker": "P40U.SI", "Sector": "REITs & Business Trusts"},
    "STARHUB": {"Rate": 0.05, "PB": 2.80, "Ticker": "CC3.SI", "Sector": "Telecommunications"},
    "UOL": {"Rate": 0.20, "PB": 0.45, "Ticker": "U14.SI", "Sector": "Real Estate"},
    "BUKIT SEMBAWANG": {"Rate": 0.01, "PB": 2.45, "Ticker": "B61.SI", "Sector": "Real Estate"},
    "HPH TRUST USD": {"Rate": 0.02, "PB": 0.35, "Ticker": "NS8U.SI", "Sector": "REITs & Business Trusts"},
    "ASTREAVIB310318": {"Rate": 0.03, "PB": 1.00, "Ticker": None, "Sector": "Fixed Income"},
    "SBDEC17 GX17120W": {"Rate": 0.03, "PB": 1.00, "Ticker": None, "Sector": "Fixed Income"}
}

# --- 4. HELPER FUNCTIONS ---
def clean_val(v):
    return float(re.sub(r'[^\d.-]', '', str(v))) if v and pd.notnull(v) and str(v).strip() != '' else 0.0

def guess_sector(name):
    name_up = str(name).upper()
    if any(x in name_up for x in ["REIT", "TRUST", "TR", "HPH", "CAPITALAND", "MAPLETREE"]): return "REITs & Business Trusts"
    if any(x in name_up for x in ["BANK", "HOLDINGS", "FINANCIAL"]): return "Financial Services"
    if any(x in name_up for x in ["TECH", "SOFTWARE", "SYSTEMS"]): return "Technology"
    if any(x in name_up for x in ["ASTREA", "BOND", "SBDEC"]): return "Fixed Income"
    return "Equities (Unclassified)"

def resolve_intel(raw_name):
    """Returns (intel_dict, canonical_name, is_known).
    is_known=False means the stock was not found in MASTER_INTEL or ALIAS_MAP.
    """
    name_up = str(raw_name).upper()
    if name_up in MASTER_INTEL:
        return MASTER_INTEL[name_up], name_up, True
    ALIAS_MAP = {
        "OCBC": ["OVERSEA-CHINESE", "OVERSEA CHINESE", "OCBC"],
        "UOB": ["UNITED OVERSEAS", "UOB"],
        "DBS": ["DBS GROUP", "DBS"],
        "SINGTEL": ["SINGAPORE TELECOM", "SINGTEL"],
        "MAPLETREE LOG TR": ["MAPLETREE LOGISTICS", "MAPLETREE LOG"],
        "ST ENGINEERING": ["SINGAPORE TECHNOLOGIES", "ST ENGG"],
        "CITYDEV": ["CITY DEVELOPMENTS"],
        "COMFORTDELGRO": ["COMFORT DELGRO"]
    }
    for clean_name, aliases in ALIAS_MAP.items():
        if any(alias in name_up for alias in aliases):
            return MASTER_INTEL[clean_name], clean_name, True
    return {"Rate": 0.0, "PB": 1.0, "Ticker": None, "Sector": guess_sector(name_up)}, name_up, False

def extract_pdf(file):
    data, seen = [], set()
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for row in table:
                    row = [str(c).replace('\n', ' ').strip() for c in row if c is not None and str(c).strip() != '']
                    if len(row) < 3: continue
                    raw_name = row[0].upper()
                    if any(x in raw_name for x in ["SECURITY", "BALANCE", "DATE", "TOTAL", "PAGE", "STATEMENT", "PORTFOLIO", "SUB-TOTAL", "SECURITIES A/C", "A/C NO"]): continue
                    if re.match(r'^\d{1,2}[\s\-/]+(?:[A-Z]{3}|\d{1,2})[\s\-/]+\d{2,4}', raw_name): continue
                    try:
                        numbers = [clean_val(c) for c in row[1:] if re.search(r'\d', str(c))]
                        if len(numbers) >= 2:
                            qty, mkt = numbers[0], numbers[-1]
                            if qty < 1 or mkt <= 0: continue
                            intel, clean_name, is_known = resolve_intel(raw_name)
                            if clean_name in seen: continue
                            data.append({"Security": clean_name, "Sector": intel.get("Sector", guess_sector(clean_name)),
                                         "Quantity": qty, "AUM (SGD)": mkt, "DPS": intel["Rate"],
                                         "P/B Ratio": intel["PB"], "Ticker": intel["Ticker"], "Known": is_known})
                            seen.add(clean_name)
                    except Exception:
                        continue
    return pd.DataFrame(data)

def color_returns(val):
    if isinstance(val, (int, float)):
        if val > 0: return 'background-color: rgba(39, 174, 96, 0.15); color: #1E8449; font-weight: bold;'
        elif val < 0: return 'background-color: rgba(214, 48, 49, 0.15); color: #C0392B; font-weight: bold;'
    return ''

def kpi_card(icon, label, value):
    st.markdown(f'<div class="kpi-card"><div class="kpi-icon">{icon}</div><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)

def health_gauge(score):
    color = "#27AE60" if score >= 70 else "#F39C12" if score >= 40 else "#E74C3C"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"suffix": "/100", "font": {"size": 28, "color": color}},
        title={"text": "Portfolio Health Score", "font": {"size": 15, "color": "#555"}},
        gauge={"axis": {"range": [0, 100], "tickcolor": "#aaa"}, "bar": {"color": color, "thickness": 0.28},
               "bgcolor": "white", "borderwidth": 0,
               "steps": [{"range": [0, 40], "color": "rgba(231,76,60,0.12)"},
                          {"range": [40, 70], "color": "rgba(243,156,18,0.12)"},
                          {"range": [70, 100], "color": "rgba(39,174,96,0.12)"}],
               "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.8, "value": score}}
    ))
    fig.update_layout(height=260, margin=dict(t=40, b=0, l=20, r=20), paper_bgcolor="white")
    return fig

def project_dividends(div_series, annual_income_sgd, n_months=12):
    """
    Project upcoming dividend payments for the next n_months.

    Uses div_series for TIMING only (which calendar months dividends land in).
    Uses annual_income_sgd from MASTER_INTEL for AMOUNTS, so totals match
    the Annualised Income KPI card.

    Shows from start of the current month so same-month payments are never
    silently dropped (ex-div date vs payment date can differ by weeks).
    """
    if annual_income_sgd <= 0 or div_series is None or len(div_series) < 1:
        return []
    s = div_series.copy()
    if hasattr(s.index, 'tz') and s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=3)
    recent = s[s.index > cutoff]
    if len(recent) < 1:
        recent = s.tail(6)
    if len(recent) == 0:
        return []

    dates = recent.index.sort_values()

    # Payment frequency from median inter-payment interval
    if len(dates) >= 2:
        intervals = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        avg_interval = int(np.median(intervals))
    else:
        avg_interval = 365
    payments_per_year = max(1, round(365 / avg_interval))
    income_per_payment = annual_income_sgd / payments_per_year

    # Typical payment months: use the most recent full annual cycle
    recent_1y = recent[recent.index >= recent.index.max() - pd.DateOffset(years=1)]
    typical_months = sorted(set(recent_1y.index.month.tolist())) if len(recent_1y) >= 1 else sorted(set(recent.index.month.tolist()))
    # Guard: if history gives more months than frequency, take the most frequent ones
    if len(typical_months) > payments_per_year:
        from collections import Counter
        month_counts = Counter(recent.index.month.tolist())
        typical_months = sorted([m for m, _ in month_counts.most_common(payments_per_year)])

    # Typical pay-day per month from history (use last occurrence of that month)
    pay_day_map = {}
    for m in typical_months:
        same_month = recent[recent.index.month == m]
        pay_day_map[m] = int(same_month.index[-1].day) if not same_month.empty else 15

    # Project: include from the START of the current month so same-month
    # dividends are shown even if the historical pay-day has just passed
    now = pd.Timestamp.now()
    month_start = pd.Timestamp(now.year, now.month, 1)
    end_date = month_start + pd.DateOffset(months=n_months)
    projections = []

    for offset in range(n_months + 2):
        check = month_start + pd.DateOffset(months=offset)
        if check > end_date:
            break
        if check.month in typical_months:
            pay_day = min(pay_day_map.get(check.month, 15), 28)
            proj_date = pd.Timestamp(check.year, check.month, pay_day)
            projections.append((proj_date, income_per_payment))

    return projections

def project_portfolio_growth(current_aum, annual_savings, total_return, dividend_yield, n_years, reinvest_dividends):
    """
    Project AUM and income year by year.
    total_return / dividend_yield are decimals (e.g. 0.07, 0.047).
    When reinvesting, dividends compound into AUM (total return applied to full AUM).
    When not reinvesting, only price appreciation grows AUM.
    """
    price_appreciation = max(0.0, total_return - dividend_yield)
    aum = float(current_aum)
    rows = []
    for y in range(n_years + 1):
        income = aum * dividend_yield
        rows.append({
            'Year': y,
            'AUM (SGD)': round(aum, 0),
            'Annual Income (SGD)': round(income, 0),
            'Monthly Income (SGD)': round(income / 12, 0),
        })
        if y < n_years:
            growth_rate = total_return if reinvest_dividends else price_appreciation
            aum = max(0.0, aum * (1 + growth_rate) + annual_savings)
    return rows

@st.cache_data(ttl=3600)
def fetch_prices_batch(tickers_str, period="2y"):
    """Batch-download closing prices; returns DataFrame with tickers as columns."""
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    result = {}
    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            close_df = raw['Close']
            for ticker in tickers:
                if ticker in close_df.columns:
                    s = close_df[ticker].dropna()
                    if not s.empty:
                        result[ticker] = s
        else:
            c = raw['Close'] if 'Close' in raw.columns else raw.iloc[:, 0]
            if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
            if not c.dropna().empty:
                result[tickers[0]] = c
    except Exception:
        pass
    for ticker in [t for t in tickers if t not in result]:
        try:
            h = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            c = h['Close'].iloc[:, 0] if isinstance(h['Close'], pd.DataFrame) else h['Close']
            if not c.dropna().empty:
                result[ticker] = c
        except Exception:
            continue
    return pd.DataFrame(result) if result else pd.DataFrame()

@st.cache_data(ttl=3600)
def compute_risk_metrics(tickers_str, names_str, rf_annual=0.037):
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    names = [n.strip() for n in names_str.split("|") if n.strip()]
    prices = fetch_prices_batch(tickers_str, period="2y")
    try:
        sti_raw = yf.download("^STI", period="2y", auto_adjust=True, progress=False)
        sti = sti_raw['Close'].iloc[:, 0] if isinstance(sti_raw['Close'], pd.DataFrame) else sti_raw['Close']
        sti_ret = sti.pct_change().dropna()
    except Exception:
        sti_ret = pd.Series(dtype=float)
    results = []
    for ticker, name in zip(tickers, names):
        try:
            if ticker not in prices.columns or prices[ticker].dropna().shape[0] < 30:
                continue
            price = prices[ticker].dropna()
            ret = price.pct_change().dropna()
            vol = float(ret.std() * np.sqrt(252))
            ann_ret = float((1 + ret.mean()) ** 252 - 1)
            sharpe = (ann_ret - rf_annual) / vol if vol > 0 else 0.0
            beta = None
            if not sti_ret.empty:
                common = ret.index.intersection(sti_ret.index)
                if len(common) > 30:
                    cov_m = np.cov(ret.loc[common].values, sti_ret.loc[common].values)
                    beta = round(float(cov_m[0, 1] / cov_m[1, 1]), 3) if cov_m[1, 1] != 0 else None
            var95 = float(np.percentile(ret, 5)) * 100
            cumret = (1 + ret).cumprod()
            max_dd = float(((cumret - cumret.cummax()) / cumret.cummax()).min()) * 100
            results.append({"Security": name, "Ticker": ticker,
                             "Ann. Volatility (%)": round(vol * 100, 2),
                             "Ann. Return (%)": round(ann_ret * 100, 2),
                             "Sharpe Ratio": round(sharpe, 3),
                             "Beta (vs STI)": beta,
                             "Daily VaR 95% (%)": round(var95, 2),
                             "Max Drawdown (%)": round(max_dd, 2)})
        except Exception:
            continue
    return pd.DataFrame(results)

@st.cache_data(ttl=3600)
def get_dividend_data(tickers_str, names_str):
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    names = [n.strip() for n in names_str.split("|") if n.strip()]
    result = {}
    for ticker, name in zip(tickers, names):
        try:
            divs = yf.Ticker(ticker).dividends
            if not divs.empty:
                result[name] = divs
        except Exception:
            continue
    return result

@st.cache_data(ttl=3600)
def compute_efficient_frontier(tickers_str, names_str, weights_str, rf_annual=0.037, n_sim=3000):
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    names = [n.strip() for n in names_str.split("|") if n.strip()]
    curr_w_raw = [float(w) for w in weights_str.split(",") if w.strip()]
    if len(tickers) < 2:
        return None
    prices = fetch_prices_batch(tickers_str, period="3y")
    if prices.empty:
        return None
    valid_tickers = [t for t in tickers if t in prices.columns and prices[t].notna().sum() > 90]
    if len(valid_tickers) < 2:
        return None
    idx_map = {t: i for i, t in enumerate(tickers)}
    valid_names = [names[idx_map[t]] for t in valid_tickers]
    raw_w = [curr_w_raw[idx_map[t]] for t in valid_tickers]
    total_w = sum(raw_w)
    valid_curr_w = [w / total_w for w in raw_w] if total_w > 0 else [1/len(valid_tickers)] * len(valid_tickers)
    p = prices[valid_tickers].ffill().dropna()
    if len(p) < 60:
        return None
    ret = p.pct_change().dropna()
    mean_r = ret.mean().values * 252
    cov = ret.cov().values * 252
    np.random.seed(42)
    mc_r, mc_v, mc_s, mc_w = [], [], [], []
    for _ in range(n_sim):
        w = np.random.dirichlet(np.ones(len(valid_tickers)))
        r = float(np.dot(w, mean_r))
        v = float(np.sqrt(max(float(np.dot(w, np.dot(cov, w))), 0)))
        s = (r - rf_annual) / v if v > 0 else 0.0
        mc_r.append(r); mc_v.append(v); mc_s.append(s); mc_w.append(w.tolist())
    wc = np.array(valid_curr_w)
    cr = float(np.dot(wc, mean_r))
    cv = float(np.sqrt(max(float(np.dot(wc, np.dot(cov, wc))), 0)))
    cs = (cr - rf_annual) / cv if cv > 0 else 0.0
    ms_i, mv_i = int(np.argmax(mc_s)), int(np.argmin(mc_v))
    return {"names": valid_names, "mc_rets": mc_r, "mc_vols": mc_v, "mc_sharpes": mc_s, "mc_ws": mc_w,
            "curr": {"ret": cr, "vol": cv, "sharpe": cs, "weights": valid_curr_w},
            "max_sharpe": {"ret": mc_r[ms_i], "vol": mc_v[ms_i], "sharpe": mc_s[ms_i], "weights": mc_w[ms_i]},
            "min_vol": {"ret": mc_r[mv_i], "vol": mc_v[mv_i], "sharpe": mc_s[mv_i], "weights": mc_w[mv_i]}}

@st.cache_data(ttl=3600)
def scan_portfolio_returns(_data_hash, tickers_and_names):
    m = {}
    for ticker, name in tickers_and_names:
        try:
            h = yf.download(ticker, period="5y", auto_adjust=True, progress=False)
            if not h.empty:
                c = h['Close'].iloc[:, 0] if isinstance(h['Close'], pd.DataFrame) else h['Close']
                m[name] = float(c.iloc[-1]) / float(c.iloc[0]) - 1.0
        except Exception:
            continue
    return m

@st.cache_data(ttl=3600)
def load_benchmark(ticker, years):
    start = datetime.now() - timedelta(days=years * 365)
    h_a = yf.download(ticker, start=start, end=datetime.now(), auto_adjust=True, progress=False)
    h_s = yf.download("^STI", start=start, end=datetime.now(), auto_adjust=True, progress=False)
    return h_a, h_s

CRISIS_SCENARIOS = {
    "COVID Crash 2020": {
        "start": "2020-02-19", "end": "2020-03-23",
        "desc": "COVID-19 pandemic selloff — 33-day crash",
        "sti_drop": -0.305,
    },
    "GFC 2008–09": {
        "start": "2007-10-09", "end": "2009-03-09",
        "desc": "Global Financial Crisis — peak to trough (~17 months)",
        "sti_drop": -0.617,
    },
    "2022 Rate Hike Cycle": {
        "start": "2022-01-03", "end": "2022-10-31",
        "desc": "Aggressive Fed rate hike cycle — REIT and bond selloff",
        "sti_drop": -0.118,
    },
    "Asia Financial Crisis 1997–98": {
        "start": "1997-07-01", "end": "1998-08-31",
        "desc": "Asian currency crisis — regional contagion",
        "sti_drop": -0.548,
    },
}

@st.cache_data(ttl=86400)
def fetch_crisis_returns(tickers_str, start_date, end_date):
    """Returns {ticker: pct_return} over the given crisis window."""
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    result = {}
    if not tickers:
        return result
    try:
        raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            close_df = raw['Close']
            for ticker in tickers:
                if ticker in close_df.columns:
                    s = close_df[ticker].dropna()
                    if len(s) >= 2:
                        result[ticker] = float(s.iloc[-1]) / float(s.iloc[0]) - 1.0
        else:
            c = raw['Close'] if 'Close' in raw.columns else raw.iloc[:, 0]
            if isinstance(c, pd.DataFrame):
                c = c.iloc[:, 0]
            s = c.dropna()
            if len(s) >= 2:
                result[tickers[0]] = float(s.iloc[-1]) / float(s.iloc[0]) - 1.0
    except Exception:
        pass
    for ticker in [t for t in tickers if t not in result]:
        try:
            h = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
            c = h['Close'].iloc[:, 0] if isinstance(h['Close'], pd.DataFrame) else h['Close']
            s = c.dropna()
            if len(s) >= 2:
                result[ticker] = float(s.iloc[-1]) / float(s.iloc[0]) - 1.0
        except Exception:
            continue
    return result

@st.cache_data(ttl=3600)
def fetch_annual_dps(ticker):
    """Sum last 12 months of dividends from Yahoo Finance for an unknown stock."""
    if not ticker or not ticker.strip():
        return 0.0
    try:
        divs = yf.Ticker(ticker.strip()).dividends
        if divs.empty:
            return 0.0
        if hasattr(divs.index, 'tz') and divs.index.tz is not None:
            divs.index = divs.index.tz_localize(None)
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=1)
        trailing = divs[divs.index >= cutoff]
        if trailing.empty:
            trailing = divs.tail(4)
        return round(float(trailing.sum()), 4)
    except Exception:
        return 0.0

# --- 5. SIDEBAR ---
st.sidebar.markdown("""
<div style='text-align:center; padding:16px 0 22px 0;'>
  <div style='font-size:2.4rem;'>🏦</div>
  <div style='font-size:1.05rem; font-weight:700; color:#f0b429; letter-spacing:1.5px; margin-top:6px;'>CLP WEALTH CENTER</div>
  <div style='font-size:0.72rem; color:#8fa8cc; margin-top:4px; letter-spacing:0.5px;'>Portfolio Intelligence Platform</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("**Upload Portfolio Statement**")
uploaded_file = st.sidebar.file_uploader("PDF only", type="pdf", label_visibility="collapsed")
st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ App Maintenance Guide"):
    st.markdown("""
**The Golden Rule of New Stocks**

If you purchase a brand new asset, the app calculates AUM automatically but hides it from Market Benchmark and Audit tabs.

**To unlock full auditing:**
1. Open `app.py` in your GitHub repository.
2. Locate the `MASTER_INTEL` dict near the top.
3. Add your stock:
`"PANUNITED": {"Rate": 0.0, "PB": 1.0, "Ticker": "P52.SI", "Sector": "Industrials"}`
4. Click **Commit changes** — the live app refreshes instantly.
""")

# --- 6. TABS ---
t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs([
    "📊 Discovery", "📈 Benchmark", "📂 Holdings",
    "📜 Advisory", "⚠️ Risk", "📅 Dividends",
    "🎯 Optimise", "🔥 Stress Test", "🏆 Goals"
])

# --- 7. MAIN CONTENT ---
if uploaded_file:
    with st.spinner("Parsing portfolio statement — please wait..."):
        df_raw = extract_pdf(uploaded_file)

    if df_raw.empty:
        st.error("Could not extract any holdings from this PDF. Please check the file format.")
    else:
        all_sects = sorted(df_raw['Sector'].unique())
        sel_sects = st.sidebar.multiselect("Filter Analysis Universe", all_sects, default=all_sects)
        df = df_raw[df_raw['Sector'].isin(sel_sects)].copy()

        # ── Unknown-stock supplemental data UI ───────────────────────
        unknown_stocks = df[df['Known'] == False]['Security'].tolist() if 'Known' in df.columns else []
        if unknown_stocks:
            with st.sidebar.expander(f"🔍 Enrich {len(unknown_stocks)} Unknown Stock(s)", expanded=True):
                st.markdown(
                    "<small>These holdings were not found in the built-in database. "
                    "Enter a Yahoo Finance ticker (e.g. <code>BN4.SI</code>) to fetch live DPS data, "
                    "or enter a DPS manually.</small>",
                    unsafe_allow_html=True,
                )
                for stk in unknown_stocks:
                    st.markdown(f"**{stk}**")
                    col_t, col_d = st.columns([3, 2])
                    ticker_key = f"supp_ticker_{stk}"
                    dps_key = f"supp_dps_{stk}"
                    fetch_key = f"supp_fetched_{stk}"
                    with col_t:
                        ticker_input = st.text_input("Ticker", key=ticker_key,
                                                     placeholder="e.g. BN4.SI",
                                                     label_visibility="collapsed")
                    if ticker_input and ticker_input != st.session_state.get(fetch_key, ""):
                        auto_dps = fetch_annual_dps(ticker_input)
                        st.session_state[dps_key] = auto_dps
                        st.session_state[fetch_key] = ticker_input
                        # also store ticker for downstream tab use
                        st.session_state[f"supp_ticker_val_{stk}"] = ticker_input
                    with col_d:
                        dps_default = st.session_state.get(dps_key, 0.0)
                        dps_override = st.number_input("DPS (S$)", key=f"supp_dps_input_{stk}",
                                                        min_value=0.0, step=0.01,
                                                        value=float(dps_default),
                                                        label_visibility="collapsed")
                        st.session_state[dps_key] = dps_override

        # Apply session_state DPS overrides and ticker enrichment to df
        if 'Known' in df.columns:
            for stk in df[df['Known'] == False]['Security'].tolist():
                dps_val = st.session_state.get(f"supp_dps_input_{stk}", st.session_state.get(f"supp_dps_{stk}", 0.0))
                ticker_val = st.session_state.get(f"supp_ticker_val_{stk}", None)
                mask = df['Security'] == stk
                df.loc[mask, 'DPS'] = float(dps_val)
                if ticker_val:
                    df.loc[mask, 'Ticker'] = ticker_val

        total_aum = df['AUM (SGD)'].sum()
        df["Annual Dividend (SGD)"] = (df["Quantity"] * df["DPS"]).round(2)
        total_inc = df['Annual Dividend (SGD)'].sum()
        df["Dividend Yield (%)"] = (df["Annual Dividend (SGD)"] / df["AUM (SGD)"] * 100).round(2)
        df["%Portfolio"] = (df["AUM (SGD)"] / total_aum * 100).round(2) if total_aum > 0 else 0
        df["%Income"] = (df["Annual Dividend (SGD)"] / total_inc * 100).round(2) if total_inc > 0 else 0
        df["Alloc_Label"] = df.apply(lambda r: f"{r['Security']} ({r['%Portfolio']:.1f}%)", axis=1)
        df["Inc_Label"] = df.apply(lambda r: f"{r['Security']} ({r['%Income']:.1f}%)", axis=1)
        port_yield = (total_inc / total_aum * 100) if total_aum > 0 else 0

        sector_pct = df.groupby('Sector')["%Portfolio"].sum()
        top_sector_pct = sector_pct.max()
        if top_sector_pct > 40:
            st.markdown(f'<div class="conc-alert">⚠️ <strong>Concentration Alert:</strong> <em>{sector_pct.idxmax()}</em> makes up <strong>{top_sector_pct:.1f}%</strong> of your portfolio. Consider diversifying to reduce sector-specific risk.</div>', unsafe_allow_html=True)

        k1, k2, k3 = st.columns(3)
        with k1: kpi_card("💼", "Total AUM", f"S${total_aum:,.0f}")
        with k2: kpi_card("💰", "Annualised Income", f"S${total_inc:,.0f}")
        with k3: kpi_card("📈", "Portfolio Yield", f"{port_yield:.2f}%")

        # ── DPS data-quality banner ───────────────────────────────────
        active_notes = {k: v for k, v in DPS_NOTES.items() if k in df['Security'].values}
        note_items = "".join(
            f"<li style='margin-bottom:4px;'><strong>{k}</strong> — {v}</li>"
            for k, v in active_notes.items()
        )
        # Warn about unknown stocks still carrying DPS=0
        if 'Known' in df.columns:
            zero_dps_unknown = df[(df['Known'] == False) & (df['DPS'] == 0)]['Security'].tolist()
            if zero_dps_unknown:
                names_str = ", ".join(zero_dps_unknown)
                note_items += (
                    f"<li style='margin-bottom:4px;color:#C0392B;'><strong>Missing DPS data:</strong> "
                    f"{names_str} — income is shown as S$0. Use the sidebar enrichment panel to add a ticker or enter DPS manually.</li>"
                )
        note_block = f"<ul style='margin:6px 0 2px 0;padding-left:20px;'>{note_items}</ul>" if note_items else ""
        st.markdown(
            f"<div style='background:rgba(243,156,18,0.07);border-left:4px solid #F39C12;"
            f"border-radius:0 8px 8px 0;padding:11px 16px;margin:14px 0 4px 0;"
            f"font-size:0.84rem;color:#7D6608;'>"
            f"⚠️ <strong>Dividend Data Notice</strong> — DPS values reflect <strong>{_DPS_AS_OF}</strong> "
            f"filings. They are manually maintained and must be updated whenever companies "
            f"announce dividend changes. <em>Income figures are estimates, not guarantees.</em>"
            f"{note_block}</div>",
            unsafe_allow_html=True
        )

        # ══════════════════════════════════════════
        # TAB 1 — ASSET DISCOVERY
        # ══════════════════════════════════════════
        with t1:
            st.markdown('<p class="tab-subtitle">Visualise how your capital and income are distributed across sectors and individual holdings.</p>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<h4 style='text-align:center;'>Capital Allocation</h4>", unsafe_allow_html=True)
                fig_tree = px.treemap(df, path=['Sector', 'Alloc_Label'], values='AUM (SGD)', color='Sector')
                fig_tree.update_layout(margin=dict(t=10, l=10, r=10, b=10))
                st.plotly_chart(fig_tree, use_container_width=True)
            with c2:
                st.markdown("<h4 style='text-align:center;'>Income Stream</h4>", unsafe_allow_html=True)
                fig_sun = px.sunburst(df, path=['Sector', 'Inc_Label'], values='Annual Dividend (SGD)', color='Sector')
                fig_sun.update_layout(margin=dict(t=10, l=10, r=10, b=10))
                st.plotly_chart(fig_sun, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h4 style='text-align:center;'>Asset Efficiency Matrix: Valuation vs. Yield</h4>", unsafe_allow_html=True)
            fig_scatter = px.scatter(df[df['Sector'] != 'Fixed Income'], x="P/B Ratio", y="Dividend Yield (%)",
                                     size="AUM (SGD)", color="Sector", hover_name="Security", template="plotly_dark", height=500)
            fig_scatter.update_layout(margin=dict(t=20, l=10, r=10, b=10))
            st.plotly_chart(fig_scatter, use_container_width=True)

        # ══════════════════════════════════════════
        # TAB 2 — MARKET BENCHMARK
        # ══════════════════════════════════════════
        with t2:
            st.markdown('<p class="tab-subtitle">Compare any holding\'s total return against the STI benchmark index over your chosen time horizon.</p>', unsafe_allow_html=True)
            eligible = df[df['Ticker'].notnull()]['Security'].unique()
            if len(eligible) > 0:
                c_s, c_i = st.columns(2)
                target = c_s.selectbox("Select Asset", options=eligible)
                yrs = c_i.selectbox("Time Horizon", [1, 5, 10, 15, 20], format_func=lambda x: f"{x} Year{'s' if x > 1 else ''}")
                with st.spinner(f"Loading {yrs}-year data for {target} vs STI..."):
                    tick = df[df['Security'] == target]['Ticker'].iloc[0]
                    h_a, h_s = load_benchmark(tick, yrs)
                if not h_a.empty and not h_s.empty:
                    c_a = h_a['Close'].iloc[:, 0] if isinstance(h_a['Close'], pd.DataFrame) else h_a['Close']
                    c_sti = h_s['Close'].iloc[:, 0] if isinstance(h_s['Close'], pd.DataFrame) else h_s['Close']
                    asset_norm = (c_a / float(c_a.iloc[0])) * 100
                    sti_norm = (c_sti / float(c_sti.iloc[0])) * 100
                    final_asset, final_sti = float(asset_norm.iloc[-1]), float(sti_norm.iloc[-1])
                    bm1, bm2, bm3 = st.columns(3)
                    bm1.metric(f"{target} Return", f"{final_asset - 100:+.1f}%")
                    bm2.metric("STI Return", f"{final_sti - 100:+.1f}%")
                    bm3.metric("Alpha vs STI", f"{final_asset - final_sti:+.1f}%", delta_color="normal")
                    fig_bm = go.Figure()
                    fig_bm.add_trace(go.Scatter(x=asset_norm.index, y=asset_norm, name=target,
                                                line=dict(color='#27AE60', width=2.5), fill='tozeroy', fillcolor='rgba(39,174,96,0.06)'))
                    fig_bm.add_trace(go.Scatter(x=sti_norm.index, y=sti_norm, name="STI Benchmark",
                                                line=dict(color='#95A5A6', width=2, dash='dot')))
                    fig_bm.add_hline(y=100, line_dash="solid", line_color="#ccc", line_width=1)
                    fig_bm.update_layout(template="plotly_white", height=420, yaxis_title="Indexed Return (Base = 100)",
                                         legend=dict(orientation="h", y=1.08), margin=dict(t=20, l=10, r=10, b=10))
                    st.plotly_chart(fig_bm, use_container_width=True)
                else:
                    st.warning("Could not retrieve market data for this asset. Try a different selection.")
            else:
                st.warning("No benchmarkable assets found in the current selection.")

        # ══════════════════════════════════════════
        # TAB 3 — VERIFICATION HUB
        # ══════════════════════════════════════════
        with t3:
            SECTOR_ICONS = {
                "Financial Services": "🏦", "REITs & Business Trusts": "🏢",
                "Telecommunications": "📡", "Technology": "💻",
                "Industrials & Diversified": "⚙️", "Consumer Goods": "🛒",
                "Real Estate": "🏗️", "Fixed Income": "📄",
                "Equities (Unclassified)": "📊",
            }

            def _yield_color(val):
                if not isinstance(val, (int, float)): return ''
                if val >= 5.0: return 'background-color:rgba(39,174,96,0.18);color:#1E8449;font-weight:600;'
                if val >= 3.0: return 'background-color:rgba(243,156,18,0.13);color:#7D6608;font-weight:500;'
                if val > 0:    return 'background-color:rgba(214,48,49,0.12);color:#C0392B;'
                return ''

            def _pct_color(val):
                if isinstance(val, (int, float)) and val > 0:
                    return f'background-color:rgba(41,128,185,{min(0.45, val/100*4.5):.2f});'
                return ''

            st.markdown('<p class="tab-subtitle">Audit every holding across sectors — AUM, DPS, yield, and income contribution in one place.</p>', unsafe_allow_html=True)

            # ── Top summary row ──────────────────────────────────────
            vh1, vh2, vh3, vh4 = st.columns(4)
            top_sector = df.groupby('Sector')['AUM (SGD)'].sum().idxmax() if not df.empty else "—"
            top_sector_label = (top_sector[:16] + "…") if len(top_sector) > 17 else top_sector
            with vh1: kpi_card("📋", "Total Holdings", str(len(df)))
            with vh2: kpi_card("🗂️", "Sectors", str(df['Sector'].nunique()))
            with vh3: kpi_card("🏆", "Largest Sector", top_sector_label)
            with vh4: kpi_card("📈", "Portfolio Yield", f"{port_yield:.2f}%")

            _, dl_col = st.columns([3, 1])
            with dl_col:
                csv_data = df[['Security', 'Sector', 'Quantity', 'DPS', 'Annual Dividend (SGD)', 'Dividend Yield (%)', '%Portfolio', 'AUM (SGD)']].to_csv(index=False)
                st.download_button("📥 Export Holdings to CSV", data=csv_data,
                                   file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                                   mime="text/csv", use_container_width=True)

            # ── Sector overview chart ────────────────────────────────
            st.markdown("---")
            st.markdown("#### Sector Overview")
            sector_ov = df.groupby('Sector').agg(
                AUM=('AUM (SGD)', 'sum'),
                Holdings=('Security', 'count'),
                Income=('Annual Dividend (SGD)', 'sum'),
            ).reset_index().sort_values('AUM', ascending=False)
            sector_ov['Yield (%)'] = (sector_ov['Income'] / sector_ov['AUM'] * 100).round(2)
            sector_ov['% Portfolio'] = (sector_ov['AUM'] / total_aum * 100).round(1)
            max_yield_ov = max(float(sector_ov['Yield (%)'].max()), 6.0)
            fig_ov = go.Figure(go.Bar(
                y=sector_ov['Sector'],
                x=sector_ov['AUM'],
                orientation='h',
                marker=dict(
                    color=sector_ov['Yield (%)'],
                    colorscale=[[0, '#1a3a6b'], [0.45, '#2980B9'], [1.0, '#f0b429']],
                    cmin=0, cmax=max_yield_ov,
                    colorbar=dict(title='Yield %', thickness=13, len=0.85, tickfont=dict(size=10)),
                    line=dict(color='white', width=0.5),
                ),
                text=[f"  {p:.1f}%  ·  {h} holding{'s' if h > 1 else ''}"
                      for p, h in zip(sector_ov['% Portfolio'], sector_ov['Holdings'])],
                textposition='inside',
                textfont=dict(color='white', size=11),
                hovertemplate='<b>%{y}</b><br>AUM: S$%{x:,.0f}<br>Yield: %{marker.color:.2f}%<extra></extra>',
            ))
            fig_ov.update_layout(
                template='plotly_white',
                height=max(260, len(sector_ov) * 46 + 60),
                margin=dict(t=10, b=10, l=10, r=80),
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(autorange='reversed', tickfont=dict(size=12, color='#333')),
                paper_bgcolor='white', plot_bgcolor='white',
            )
            st.plotly_chart(fig_ov, use_container_width=True)
            st.caption("Bar length = AUM · Colour = dividend yield (navy → gold)")

            # ── Per-sector drill-down ────────────────────────────────
            st.markdown("---")
            st.markdown("#### DPS Audit — Verify Before Trusting Income Figures")
            with st.expander("🔍 Open DPS Audit Table — cross-check these values against the latest company announcements", expanded=False):
                st.markdown(
                    f"DPS values sourced from **{_DPS_AS_OF}** filings. "
                    "Holdings flagged in amber require attention before income figures can be relied upon. "
                    "To correct a value, update the `Rate` field in `MASTER_INTEL` in `app.py`."
                )
                audit_rows = []
                for _, row in df.sort_values('%Income', ascending=False).iterrows():
                    note = DPS_NOTES.get(row['Security'], "")
                    audit_rows.append({
                        "Security": row['Security'],
                        "Sector": row['Sector'],
                        "DPS Used (SGD)": row['DPS'],
                        "Quantity": row['Quantity'],
                        "Annual Income (SGD)": row['Annual Dividend (SGD)'],
                        "% of Total Income": row['%Income'],
                        "Flag": "⚠️ See note" if note else "✅ No flag",
                        "Note": note if note else "—",
                    })
                audit_df = pd.DataFrame(audit_rows)

                def _flag_style(val):
                    if isinstance(val, str) and "⚠️" in val:
                        return 'background-color:rgba(243,156,18,0.18);color:#7D6608;font-weight:600;'
                    if isinstance(val, str) and "✅" in val:
                        return 'background-color:rgba(39,174,96,0.12);color:#1E8449;'
                    return ''

                styled_audit = audit_df.style.format({
                    'DPS Used (SGD)': '{:.4f}',
                    'Quantity': '{:,.0f}',
                    'Annual Income (SGD)': 'S${:,.2f}',
                    '% of Total Income': '{:.2f}%',
                }).map(_flag_style, subset=['Flag'])
                st.dataframe(styled_audit, hide_index=True, use_container_width=True)

                flagged = audit_df[audit_df['Flag'] != "✅ No flag"]
                if not flagged.empty:
                    st.warning(
                        f"**{len(flagged)} holding{'s' if len(flagged) > 1 else ''} flagged** — "
                        f"these contribute **S${flagged['Annual Income (SGD)'].sum():,.2f}** "
                        f"({flagged['% of Total Income'].sum():.1f}%) of projected annual income. "
                        "Resolve flags above before using income figures for financial planning."
                    )
                else:
                    st.success("All DPS values are flagged clean for this portfolio.")

            st.markdown("---")
            st.markdown("#### Holdings by Sector")
            for s in sorted(df['Sector'].unique()):
                sdf = df[df['Sector'] == s].copy().sort_values('AUM (SGD)', ascending=False)
                sdf.insert(0, '#', range(1, len(sdf) + 1))
                sector_aum = float(sdf['AUM (SGD)'].sum())
                sector_inc = float(sdf['Annual Dividend (SGD)'].sum())
                sector_yield = (sector_inc / sector_aum * 100) if sector_aum > 0 else 0.0
                sector_pct = (sector_aum / total_aum * 100)
                icon = SECTOR_ICONS.get(s, "📌")

                with st.expander(
                    f"{icon}  {s}   ·   {len(sdf)} holding{'s' if len(sdf) > 1 else ''}   ·   "
                    f"S${sector_aum:,.0f}   ·   {sector_pct:.1f}% of portfolio   ·   avg yield {sector_yield:.2f}%"
                ):
                    sk1, sk2, sk3 = st.columns(3)
                    with sk1: kpi_card("💼", "Sector AUM", f"S${sector_aum:,.0f}")
                    with sk2: kpi_card("💰", "Annual Income", f"S${sector_inc:,.0f}")
                    with sk3: kpi_card("📈", "Avg Yield", f"{sector_yield:.2f}%")

                    st.markdown("<br>", unsafe_allow_html=True)
                    ch_col, tb_col = st.columns([1, 1.15])

                    with ch_col:
                        max_y = max(float(sdf['Dividend Yield (%)'].max()), 3.0)
                        fig_s = go.Figure(go.Bar(
                            x=sdf['AUM (SGD)'],
                            y=sdf['Security'],
                            orientation='h',
                            marker=dict(
                                color=sdf['Dividend Yield (%)'],
                                colorscale=[[0, '#1a3a6b'], [0.45, '#2980B9'], [1.0, '#f0b429']],
                                cmin=0, cmax=max_y,
                                showscale=False,
                                line=dict(color='white', width=0.5),
                            ),
                            text=[f"S${v:,.0f}" for v in sdf['AUM (SGD)']],
                            textposition='outside',
                            textfont=dict(size=10, color='#333'),
                            hovertemplate='<b>%{y}</b><br>AUM: S$%{x:,.0f}<br>Yield: %{marker.color:.2f}%<extra></extra>',
                        ))
                        fig_s.update_layout(
                            template='plotly_white',
                            height=max(180, len(sdf) * 40 + 50),
                            margin=dict(t=6, b=6, l=6, r=70),
                            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                            yaxis=dict(autorange='reversed', tickfont=dict(size=11, color='#333')),
                            paper_bgcolor='white', plot_bgcolor='#fafbfd',
                        )
                        st.plotly_chart(fig_s, use_container_width=True)

                    with tb_col:
                        disp_cols = ['#', 'Security', 'Quantity', 'DPS', 'Annual Dividend (SGD)', 'Dividend Yield (%)', '%Portfolio', 'AUM (SGD)']
                        styled_sdf = sdf[disp_cols].style.format({
                            'DPS': '{:.2f}',
                            'Annual Dividend (SGD)': 'S${:,.0f}',
                            'Dividend Yield (%)': '{:.2f}%',
                            '%Portfolio': '{:.1f}%',
                            'AUM (SGD)': 'S${:,.0f}',
                        }).map(_yield_color, subset=['Dividend Yield (%)']).map(_pct_color, subset=['%Portfolio'])
                        st.dataframe(styled_sdf, hide_index=True, use_container_width=True)

        # ══════════════════════════════════════════
        # TAB 4 — STRATEGIC ADVISORY
        # ══════════════════════════════════════════
        with t4:
            st.markdown('<p class="tab-subtitle">Audit your portfolio health, identify wealth creators and destroyers, and simulate rebalancing scenarios.</p>', unsafe_allow_html=True)
            eligible_df = df[df['Ticker'].notnull()]
            tickers_and_names = list(zip(eligible_df['Ticker'], eligible_df['Security']))
            data_hash = str(sorted([t for t, _ in tickers_and_names]))
            p_map = scan_portfolio_returns(data_hash, tickers_and_names)
            losers = [k for k, v in p_map.items() if float(v) < 0.0]
            winners = sorted([k for k, v in p_map.items() if float(v) >= 0.0], key=lambda x: float(p_map[x]), reverse=True)[:10]
            health = max(100 - (len(losers) * 5), 0)
            h_col1, h_col2 = st.columns([1, 1])
            with h_col1:
                st.plotly_chart(health_gauge(health), use_container_width=True)
            with h_col2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                top_divs = df.nlargest(2, 'Annual Dividend (SGD)')['Security'].tolist()
                anchor_text = f"**{', '.join(top_divs)}**" if top_divs else "diversified cash holdings"
                st.markdown(f"**Strengths:** Core income stability anchored by {anchor_text}.")
                st.markdown(f"**Priority:** Investigate **{len(losers)}** asset{'s' if len(losers) != 1 else ''} underperforming capital benchmarks over 5 years.")
                if health >= 70: st.success("Portfolio is in **good health**. Focus on growth and income optimisation.")
                elif health >= 40: st.warning("Portfolio health is **moderate**. Consider reviewing underperformers.")
                else: st.error("Portfolio health is **below target**. Action on losers is recommended.")
            st.markdown("---")
            c_aud1, c_aud2 = st.columns(2)
            with c_aud1:
                st.markdown("### 💎 Wealth Creator Audit")
                if len(winners) > 0:
                    sel_w = st.multiselect("High-Performers", options=winners, default=winners[:2])
                    wy = st.multiselect("Horizons (Years)", options=[5, 10, 15, 20], default=[5, 10], key="wy")
                    if st.button("📈 Audit Creators"):
                        res_w = []
                        for n in sel_w:
                            rd = {"Security": n}
                            for y in wy:
                                tick = df[df['Security'] == n]['Ticker'].iloc[0]
                                h = yf.download(tick, start=(datetime.now() - timedelta(days=y * 365)), end=datetime.now(), auto_adjust=True, progress=False)
                                if not h.empty:
                                    c = h['Close'].iloc[:, 0] if isinstance(h['Close'], pd.DataFrame) else h['Close']
                                    rd[f"{y}Y Total Return"] = (float(c.iloc[-1]) / float(c.iloc[0]) - 1.0) * 100
                            res_w.append(rd)
                        df_w = pd.DataFrame(res_w)
                        st.dataframe(df_w.style.format({col: "{:.2f}%" for col in df_w.columns if "Return" in col}, na_rep="-").map(color_returns), use_container_width=True, hide_index=True)
                else:
                    st.info("No positive-returning assets found in the current selection.")
            with c_aud2:
                st.markdown("### 🚨 Wealth Destroyer Audit")
                if len(losers) > 0:
                    sel_l = st.multiselect("Suspects", options=losers)
                    ly = st.multiselect("Horizons (Years)", options=[5, 10, 15, 20], default=[5, 10], key="ly")
                    if st.button("🔍 Audit Destroyers"):
                        res_l = []
                        for n in sel_l:
                            rd = {"Security": n}
                            for y in ly:
                                tick = df[df['Security'] == n]['Ticker'].iloc[0]
                                h = yf.download(tick, start=(datetime.now() - timedelta(days=y * 365)), end=datetime.now(), auto_adjust=True, progress=False)
                                if not h.empty:
                                    c = h['Close'].iloc[:, 0] if isinstance(h['Close'], pd.DataFrame) else h['Close']
                                    rd[f"{y}Y Total Return"] = (float(c.iloc[-1]) / float(c.iloc[0]) - 1.0) * 100
                            res_l.append(rd)
                        df_l = pd.DataFrame(res_l)
                        st.dataframe(df_l.style.format({col: "{:.2f}%" for col in df_l.columns if "Return" in col}, na_rep="-").map(color_returns), use_container_width=True, hide_index=True)
                else:
                    st.success("No wealth-destroying assets detected!")
            st.caption("*High Performers and Suspects are classified based on 5-Year Total Return.*")
            st.markdown("---")
            st.markdown("### ⚖️ Portfolio Rebalance Simulator")
            st.markdown('<p class="tab-subtitle" style="margin-top:-10px;">Model the income and yield impact of rotating capital between holdings before committing.</p>', unsafe_allow_html=True)
            col_sell, col_buy = st.columns(2)
            with col_sell:
                st.error("📉 Source of Funds — Sell")
                held_losers = [l for l in losers if l in df['Security'].values]
                sim_sell = st.multiselect("Select assets to liquidate", options=df['Security'].unique(), default=held_losers)
            with col_buy:
                st.success("📈 Destination of Funds — Buy")
                sim_buy = st.multiselect("Select assets to acquire", options=df['Security'].unique(), default=winners[:2] if len(winners) > 1 else None)
            if st.button("🔄 Execute Simulation"):
                if not sim_sell or not sim_buy:
                    st.warning("⚠️ Please select at least one asset to sell and one to buy.")
                else:
                    curr_aum = total_aum
                    curr_inc = df['Annual Dividend (SGD)'].sum()
                    curr_yield = (curr_inc / curr_aum) * 100 if curr_aum > 0 else 0
                    df_sell = df[df['Security'].isin(sim_sell)]
                    freed_capital = df_sell['AUM (SGD)'].sum()
                    lost_income = df_sell['Annual Dividend (SGD)'].sum()
                    new_income_added = 0
                    capital_per_buy = freed_capital / len(sim_buy)
                    with st.spinner("Fetching live market yields for acquisition targets..."):
                        for b in sim_buy:
                            tick = df[df['Security'] == b]['Ticker'].iloc[0]
                            dps = df[df['Security'] == b]['DPS'].iloc[0]
                            try:
                                if pd.notnull(tick):
                                    h = yf.download(tick, period="5d", progress=False)
                                    c = h['Close'].iloc[:, 0] if isinstance(h['Close'], pd.DataFrame) else h['Close']
                                    new_income_added += capital_per_buy * (dps / float(c.iloc[-1]))
                                else:
                                    new_income_added += capital_per_buy * 0.04
                            except Exception:
                                new_income_added += capital_per_buy * 0.04
                    new_aum = curr_aum
                    new_inc = curr_inc - lost_income + new_income_added
                    new_yield = (new_inc / new_aum) * 100
                    st.success(f"**Simulation Complete:** S${freed_capital:,.2f} recycled from {len(sim_sell)} asset{'s' if len(sim_sell) != 1 else ''} into {len(sim_buy)} asset{'s' if len(sim_buy) != 1 else ''}.")
                    st.markdown('<div class="hide-delta">', unsafe_allow_html=True)
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Projected AUM (Excl. Fees)", f"S${new_aum:,.2f}", "S$0.00 (Neutral)")
                    r2.metric("Projected Annual Income", f"S${new_inc:,.2f}", f"S${new_inc - curr_inc:+,.2f} / year")
                    r3.metric("Projected Portfolio Yield", f"{new_yield:.2f}%", f"{new_yield - curr_yield:+.2f}% shift")
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("#### Before vs After Comparison")
                    fig_ba = go.Figure()
                    fig_ba.add_trace(go.Bar(name="Current", x=["Annual Income (SGD)", "Portfolio Yield (%)"],
                                            y=[curr_inc, curr_yield], marker_color="#95A5A6",
                                            text=[f"S${curr_inc:,.0f}", f"{curr_yield:.2f}%"], textposition="outside"))
                    fig_ba.add_trace(go.Bar(name="Projected", x=["Annual Income (SGD)", "Portfolio Yield (%)"],
                                            y=[new_inc, new_yield], marker_color="#27AE60",
                                            text=[f"S${new_inc:,.0f}", f"{new_yield:.2f}%"], textposition="outside"))
                    fig_ba.update_layout(barmode="group", template="plotly_white", height=360,
                                         legend=dict(orientation="h", y=1.1), margin=dict(t=30, b=20, l=20, r=20),
                                         yaxis=dict(showticklabels=False, showgrid=False))
                    st.plotly_chart(fig_ba, use_container_width=True)

        # ══════════════════════════════════════════
        # TAB 5 — RISK ANALYTICS
        # ══════════════════════════════════════════
        with t5:
            st.markdown('<p class="tab-subtitle">Quantitative risk metrics per holding — volatility, Sharpe ratio, beta vs STI, Value at Risk, and maximum drawdown.</p>', unsafe_allow_html=True)
            eligible_r = df[df['Ticker'].notnull()]
            if eligible_r.empty:
                st.warning("No holdings with market tickers available. Add tickers to MASTER_INTEL to enable this tab.")
            else:
                tickers_str_r = ",".join(eligible_r['Ticker'].tolist())
                names_str_r = "|".join(eligible_r['Security'].tolist())
                with st.spinner("Computing risk metrics from 2-year price history..."):
                    risk_df = compute_risk_metrics(tickers_str_r, names_str_r)
                if risk_df.empty:
                    st.warning("Could not compute risk metrics. Check your internet connection.")
                else:
                    risk_m = risk_df.merge(eligible_r[['Security', 'AUM (SGD)']], on='Security', how='left')
                    risk_m['Dollar VaR (SGD)'] = (risk_m['Daily VaR 95% (%)'].abs() / 100 * risk_m['AUM (SGD)']).round(0)
                    total_var = risk_m['Dollar VaR (SGD)'].sum()
                    beta_vals = risk_df['Beta (vs STI)'].dropna()
                    rv1, rv2, rv3, rv4 = st.columns(4)
                    with rv1: kpi_card("🎯", "Est. Daily VaR (95%)", f"S${total_var:,.0f}")
                    with rv2: kpi_card("📉", "Avg. Ann. Volatility", f"{risk_df['Ann. Volatility (%)'].mean():.1f}%")
                    with rv3: kpi_card("⚡", "Avg. Sharpe Ratio", f"{risk_df['Sharpe Ratio'].mean():.2f}")
                    with rv4: kpi_card("📊", "Portfolio Beta", f"{beta_vals.mean():.2f}" if not beta_vals.empty else "N/A")
                    st.caption("ℹ️ VaR = estimated max 1-day loss at 95% confidence. Sharpe uses SGD T-bill ~3.7% risk-free rate. Beta measures sensitivity to STI.")
                    st.markdown("---")
                    st.markdown("#### Holdings Risk Profile")
                    disp_cols = ['Security', 'Ann. Volatility (%)', 'Ann. Return (%)', 'Sharpe Ratio', 'Beta (vs STI)', 'Daily VaR 95% (%)', 'Max Drawdown (%)']
                    styled_risk = risk_df[disp_cols].style.format(
                        {'Ann. Volatility (%)': '{:.2f}%', 'Ann. Return (%)': '{:.2f}%',
                         'Sharpe Ratio': '{:.3f}', 'Beta (vs STI)': '{:.3f}',
                         'Daily VaR 95% (%)': '{:.2f}%', 'Max Drawdown (%)': '{:.2f}%'}, na_rep="N/A"
                    ).background_gradient(subset=['Ann. Volatility (%)'], cmap='YlOrRd') \
                     .background_gradient(subset=['Sharpe Ratio'], cmap='RdYlGn') \
                     .background_gradient(subset=['Max Drawdown (%)'], cmap='YlOrRd')
                    st.dataframe(styled_risk, use_container_width=True, hide_index=True)
                    st.markdown("---")
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        st.markdown("#### Volatility Ranking")
                        vf = px.bar(risk_df.sort_values('Ann. Volatility (%)'), x='Ann. Volatility (%)', y='Security',
                                    orientation='h', color='Ann. Volatility (%)', color_continuous_scale='YlOrRd', template='plotly_white')
                        vf.update_layout(height=max(300, len(risk_df) * 26), margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
                        st.plotly_chart(vf, use_container_width=True)
                    with rc2:
                        st.markdown("#### Risk vs Return")
                        plot_df = risk_df.dropna(subset=['Ann. Volatility (%)', 'Ann. Return (%)'])
                        rrf = px.scatter(plot_df, x='Ann. Volatility (%)', y='Ann. Return (%)',
                                         color='Sharpe Ratio', hover_name='Security',
                                         color_continuous_scale='RdYlGn', template='plotly_white', height=400)
                        rrf.add_hline(y=0, line_dash="dash", line_color="#ccc")
                        rrf.update_layout(margin=dict(t=10, b=10, l=10, r=10))
                        st.plotly_chart(rrf, use_container_width=True)
                    st.markdown("#### Correlation Matrix")
                    prices_2y = fetch_prices_batch(tickers_str_r, period="2y")
                    valid_pc = [t for t in eligible_r['Ticker'].tolist() if t in prices_2y.columns]
                    if len(valid_pc) >= 2:
                        name_map = dict(zip(eligible_r['Ticker'].tolist(), eligible_r['Security'].tolist()))
                        corr = prices_2y[valid_pc].pct_change().dropna().corr()
                        corr.columns = [name_map.get(c, c) for c in corr.columns]
                        corr.index = [name_map.get(c, c) for c in corr.index]
                        corr_r = corr.round(2)
                        cf = px.imshow(corr_r.to_numpy(), x=list(corr_r.columns), y=list(corr_r.index),
                                       color_continuous_scale='RdYlGn', zmin=-1, zmax=1,
                                       aspect='auto', text_auto='.2f', template='plotly_white')
                        cf.update_layout(height=550, margin=dict(t=10, b=10, l=10, r=10))
                        st.plotly_chart(cf, use_container_width=True)
                        st.caption("Near **+1** = move together (concentrated risk). Near **−1** = natural hedge. Near **0** = uncorrelated.")

        # ══════════════════════════════════════════
        # TAB 6 — DIVIDEND CALENDAR
        # ══════════════════════════════════════════
        with t6:
            st.markdown('<p class="tab-subtitle">Project your expected income stream across the next 12 months. Amounts match the Annualised Income KPI; Yahoo Finance data is used for payment timing only.</p>', unsafe_allow_html=True)
            active_cal_notes = {k: v for k, v in DPS_NOTES.items() if k in df['Security'].values}
            cal_note_items = "".join(f"<li><strong>{k}</strong>: {v}</li>" for k, v in active_cal_notes.items())
            cal_note_block = (
                f"<br><strong>Holdings requiring DPS verification in this portfolio:</strong>"
                f"<ul style='margin:4px 0 0 0;padding-left:18px;'>{cal_note_items}</ul>"
                if cal_note_items else ""
            )
            st.markdown(
                f"<div style='background:rgba(52,152,219,0.06);border-left:4px solid #3498DB;"
                f"border-radius:0 8px 8px 0;padding:11px 16px;font-size:0.84rem;color:#1a5276;'>"
                f"ℹ️ <strong>How amounts are calculated:</strong> "
                f"Each payment = (Quantity × DPS from MASTER_INTEL) ÷ payments per year. "
                f"DPS values reflect <strong>{_DPS_AS_OF}</strong> filings — they do not update automatically. "
                f"Yahoo Finance is used only for payment <em>timing</em> (which months), never for amounts. "
                f"Projected dates are estimates; actual ex-dividend and payment dates may differ. "
                f"<strong>Do not use these projections as a substitute for official company announcements.</strong>"
                f"{cal_note_block}</div>",
                unsafe_allow_html=True
            )
            eligible_d = df[df['Ticker'].notnull()]
            if eligible_d.empty:
                st.warning("No holdings with market tickers available.")
            else:
                tickers_str_d = ",".join(eligible_d['Ticker'].tolist())
                names_str_d = "|".join(eligible_d['Security'].tolist())
                # Use MASTER_INTEL-derived annual income (same source as KPI card)
                annual_income_map = dict(zip(eligible_d['Security'], eligible_d['Annual Dividend (SGD)']))
                with st.spinner("Fetching dividend timing data from Yahoo Finance..."):
                    div_data = get_dividend_data(tickers_str_d, names_str_d)
                if not div_data:
                    st.warning("No dividend history found for current holdings.")
                else:
                    now_ts = datetime.now()
                    months = pd.date_range(start=pd.Timestamp(now_ts.year, now_ts.month, 1), periods=12, freq='MS')
                    month_labels = [m.strftime('%b %Y') for m in months]
                    cal_data = {m: {} for m in month_labels}
                    for name, div_series in div_data.items():
                        annual_inc = annual_income_map.get(name, 0.0)
                        try:
                            projs = project_dividends(div_series, annual_inc)
                            for proj_date, income in projs:
                                label = proj_date.strftime('%b %Y')
                                if label in cal_data:
                                    cal_data[label][name] = cal_data[label].get(name, 0) + income
                        except Exception:
                            continue

                    # Build DataFrame: rows = securities with timing data, cols = months
                    all_securities = list(div_data.keys())
                    cal_df = pd.DataFrame(index=all_securities, columns=month_labels, data=0.0)
                    for m in month_labels:
                        for sec, inc in cal_data[m].items():
                            if sec in cal_df.index:
                                cal_df.loc[sec, m] = inc

                    monthly_totals = cal_df.sum(axis=0)
                    total_proj_12m = float(monthly_totals.sum())
                    best_month = monthly_totals.idxmax() if total_proj_12m > 0 else "N/A"
                    best_amt = float(monthly_totals.max()) if total_proj_12m > 0 else 0
                    lean_months = int((monthly_totals == 0).sum())

                    dc1, dc2, dc3 = st.columns(3)
                    with dc1: kpi_card("📅", "Projected Income (12M)", f"S${total_proj_12m:,.0f}")
                    with dc2: kpi_card("🏆", f"Best Month — {best_month}", f"S${best_amt:,.0f}")
                    with dc3: kpi_card("⚠️", "Months with No Income", str(lean_months))

                    # ── Reconciliation: explain any gap vs total_inc ──
                    excluded_names = [n for n in eligible_d['Security'].tolist() if n not in div_data]
                    excluded_inc = sum(annual_income_map.get(n, 0) for n in excluded_names)
                    all_holdings_no_ticker = df[df['Ticker'].isnull()]['Security'].tolist()
                    no_ticker_inc = df[df['Ticker'].isnull()]['Annual Dividend (SGD)'].sum()

                    if excluded_inc > 0 or no_ticker_inc > 0:
                        gap = total_inc - total_proj_12m
                        parts = []
                        if excluded_names:
                            parts.append(f"**{', '.join(excluded_names)}** — no timing data in Yahoo Finance (S${excluded_inc:,.0f}/yr)")
                        if all_holdings_no_ticker:
                            parts.append(f"**{', '.join(all_holdings_no_ticker)}** — no market ticker in MASTER_INTEL (S${no_ticker_inc:,.0f}/yr)")
                        st.markdown(f"""
                        <div class="conc-alert" style="color:#555; border-left-color:#3498DB; background:rgba(52,152,219,0.06);">
                        📊 <strong>Reconciliation:</strong> Annualised Income KPI = <strong>S${total_inc:,.0f}</strong> &nbsp;|&nbsp;
                        Calendar total = <strong>S${total_proj_12m:,.0f}</strong> &nbsp;|&nbsp;
                        Gap = <strong>S${gap:,.0f}</strong> — attributed to: {'; '.join(parts) if parts else 'rounding differences'}.
                        </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    if total_proj_12m > 0:
                        colors = px.colors.qualitative.Set3
                        fig_cal = go.Figure()
                        for i, sec in enumerate(all_securities):
                            row = cal_df.loc[sec].tolist()
                            if sum(row) > 0:
                                fig_cal.add_trace(go.Bar(name=sec, x=month_labels, y=row,
                                                         marker_color=colors[i % len(colors)],
                                                         hovertemplate='%{fullData.name}: S$%{y:,.2f}<extra></extra>'))
                        fig_cal.update_layout(barmode='stack', template='plotly_white', height=420,
                                              xaxis_title="Month", yaxis_title="Expected Income (SGD)",
                                              legend=dict(orientation="h", y=-0.3, x=0),
                                              margin=dict(t=20, b=100, l=20, r=20))
                        st.plotly_chart(fig_cal, use_container_width=True)

                    st.markdown("#### Monthly Breakdown")
                    summary_rows = [{"Month": m,
                                     "Holdings Paying": int((cal_df[m] > 0).sum()),
                                     "Securities": ", ".join([s for s in all_securities if cal_df.loc[s, m] > 0]) or "—",
                                     "Expected Income (SGD)": round(float(monthly_totals[m]), 2)}
                                    for m in month_labels]
                    sum_df = pd.DataFrame(summary_rows)
                    st.dataframe(sum_df.style.format({"Expected Income (SGD)": "S${:,.2f}"})
                                 .background_gradient(subset=["Expected Income (SGD)"], cmap="Greens", vmin=0),
                                 hide_index=True, use_container_width=True)

        # ══════════════════════════════════════════
        # TAB 7 — PORTFOLIO OPTIMISATION
        # ══════════════════════════════════════════
        with t7:
            st.markdown('<p class="tab-subtitle">Find the theoretically optimal capital allocation using Modern Portfolio Theory and Monte Carlo simulation.</p>', unsafe_allow_html=True)
            st.warning("⚠️ **Disclaimer:** Results are based on 3-year historical price data. Past performance does not guarantee future results. This is not financial advice.")
            eligible_e = df[df['Ticker'].notnull()]
            if len(eligible_e) < 2:
                st.warning("Need at least 2 holdings with market tickers to run optimisation.")
            else:
                tickers_str_e = ",".join(eligible_e['Ticker'].tolist())
                names_str_e = "|".join(eligible_e['Security'].tolist())
                aum_list = eligible_e['AUM (SGD)'].tolist()
                total_e_aum = sum(aum_list)
                weights_str_e = ",".join([str(round(a / total_e_aum, 6)) for a in aum_list])

                def color_change(val):
                    if isinstance(val, (int, float)):
                        if val > 0.5: return 'color: #1E8449; font-weight: bold;'
                        if val < -0.5: return 'color: #C0392B; font-weight: bold;'
                    return ''

                if st.button("🚀 Run Efficient Frontier (3,000 Simulations)"):
                    with st.spinner("Downloading 3-year price history and running Monte Carlo simulations — this may take 20–40 seconds on first run..."):
                        ef = compute_efficient_frontier(tickers_str_e, names_str_e, weights_str_e)
                    if ef is None:
                        st.error("Insufficient price data to run optimisation. Ensure holdings have at least 3 months of trading history.")
                    else:
                        ef1, ef2, ef3 = st.columns(3)
                        with ef1: kpi_card("📊", "Current Sharpe Ratio", f"{ef['curr']['sharpe']:.3f}")
                        with ef2: kpi_card("⭐", "Max Sharpe (Optimal)", f"{ef['max_sharpe']['sharpe']:.3f}")
                        with ef3: kpi_card("📈", "Sharpe Improvement", f"{ef['max_sharpe']['sharpe'] - ef['curr']['sharpe']:+.3f}")
                        st.markdown("<br>", unsafe_allow_html=True)
                        fig_ef = go.Figure()
                        fig_ef.add_trace(go.Scatter(
                            x=[v * 100 for v in ef['mc_vols']], y=[r * 100 for r in ef['mc_rets']],
                            mode='markers',
                            marker=dict(color=ef['mc_sharpes'], colorscale='RdYlGn', size=4, opacity=0.45,
                                        colorbar=dict(title='Sharpe', thickness=12)),
                            name='Monte Carlo Portfolios',
                            hovertemplate='Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
                        ))
                        fig_ef.add_trace(go.Scatter(
                            x=[ef['min_vol']['vol'] * 100], y=[ef['min_vol']['ret'] * 100], mode='markers',
                            marker=dict(symbol='diamond', size=18, color='#3498DB', line=dict(color='white', width=2)),
                            name=f"Min Volatility ({ef['min_vol']['vol']*100:.1f}% vol)"
                        ))
                        fig_ef.add_trace(go.Scatter(
                            x=[ef['max_sharpe']['vol'] * 100], y=[ef['max_sharpe']['ret'] * 100], mode='markers',
                            marker=dict(symbol='star', size=22, color='#27AE60', line=dict(color='white', width=2)),
                            name=f"Max Sharpe ({ef['max_sharpe']['sharpe']:.2f})"
                        ))
                        fig_ef.add_trace(go.Scatter(
                            x=[ef['curr']['vol'] * 100], y=[ef['curr']['ret'] * 100], mode='markers',
                            marker=dict(symbol='x', size=18, color='#E74C3C', line=dict(color='#E74C3C', width=3)),
                            name=f"Current Portfolio (Sharpe: {ef['curr']['sharpe']:.2f})"
                        ))
                        fig_ef.update_layout(template='plotly_white', height=500,
                                             xaxis_title='Annualised Volatility (%)', yaxis_title='Annualised Return (%)',
                                             legend=dict(orientation='h', y=1.12), margin=dict(t=30, b=20, l=20, r=20))
                        st.plotly_chart(fig_ef, use_container_width=True)
                        st.markdown("---")
                        w1, w2 = st.columns(2)
                        with w1:
                            st.markdown("#### ⭐ Max Sharpe Portfolio")
                            ms_rows = [{"Security": n, "Current (%)": round(cw * 100, 1),
                                        "Optimal (%)": round(ow * 100, 1), "Change (pp)": round((ow - cw) * 100, 1)}
                                       for n, cw, ow in zip(ef['names'], ef['curr']['weights'], ef['max_sharpe']['weights'])]
                            ms_df = pd.DataFrame(ms_rows).sort_values("Optimal (%)", ascending=False)
                            st.dataframe(ms_df.style.format({"Current (%)": "{:.1f}%", "Optimal (%)": "{:.1f}%", "Change (pp)": "{:+.1f}"})
                                         .map(color_change, subset=["Change (pp)"]), hide_index=True, use_container_width=True)
                            st.caption(f"Expected Return: **{ef['max_sharpe']['ret']*100:.1f}%** | Volatility: **{ef['max_sharpe']['vol']*100:.1f}%**")
                        with w2:
                            st.markdown("#### 💎 Min Volatility Portfolio")
                            mv_rows = [{"Security": n, "Current (%)": round(cw * 100, 1),
                                        "Min Vol (%)": round(ow * 100, 1), "Change (pp)": round((ow - cw) * 100, 1)}
                                       for n, cw, ow in zip(ef['names'], ef['curr']['weights'], ef['min_vol']['weights'])]
                            mv_df = pd.DataFrame(mv_rows).sort_values("Min Vol (%)", ascending=False)
                            st.dataframe(mv_df.style.format({"Current (%)": "{:.1f}%", "Min Vol (%)": "{:.1f}%", "Change (pp)": "{:+.1f}"})
                                         .map(color_change, subset=["Change (pp)"]), hide_index=True, use_container_width=True)
                            st.caption(f"Expected Return: **{ef['min_vol']['ret']*100:.1f}%** | Volatility: **{ef['min_vol']['vol']*100:.1f}%**")
                else:
                    st.markdown("""
                    <div style='text-align:center; padding:40px; background:#f4f8ff; border-radius:14px; border:1px dashed #c0d8f0; margin-top:20px;'>
                      <div style='font-size:2.5rem; margin-bottom:12px;'>📐</div>
                      <p style='font-weight:600; color:#0d2240; margin-bottom:6px;'>Efficient Frontier Analysis</p>
                      <p style='color:#6b7f9e; font-size:0.88rem; max-width:480px; margin:0 auto;'>
                        Click the button above to run 3,000 Monte Carlo simulations across your holdings.
                        The analysis identifies the optimal weight allocation to maximise risk-adjusted returns
                        or minimise volatility — based on 3-year historical price data.
                      </p>
                    </div>
                    """, unsafe_allow_html=True)

        # ══════════════════════════════════════════
        # TAB 8 — STRESS TEST
        # ══════════════════════════════════════════
        with t8:
            st.markdown('<p class="tab-subtitle">Simulate how your portfolio performs under historical market crises or custom sector shocks.</p>', unsafe_allow_html=True)
            st.warning("⚠️ **Disclaimer:** Results are illustrative. Historical mode uses actual price data where available; older periods fall back to beta-adjusted STI returns. Income impact is estimated proportionally to AUM loss. This is not financial advice.")

            # ── MODE 1: Historical Crisis Replay ──
            st.markdown("### 📉 Mode 1 — Historical Crisis Replay")
            eligible_st = df[df['Ticker'].notnull()]
            scen_name = st.selectbox("Select Crisis Scenario", list(CRISIS_SCENARIOS.keys()))
            scen = CRISIS_SCENARIOS[scen_name]
            st.caption(f"**{scen['desc']}** &nbsp;·&nbsp; {scen['start']} → {scen['end']} &nbsp;·&nbsp; STI Index: **{scen['sti_drop']*100:+.1f}%**")

            if st.button("🔥 Run Crisis Simulation", key="crisis_btn"):
                if eligible_st.empty:
                    st.warning("No holdings with market tickers available.")
                else:
                    tickers_str_st = ",".join(eligible_st['Ticker'].tolist())
                    names_str_st = "|".join(eligible_st['Security'].tolist())
                    with st.spinner(f"Fetching {scen_name} price data and computing impact..."):
                        crisis_rets = fetch_crisis_returns(tickers_str_st, scen['start'], scen['end'])
                        try:
                            risk_df_st = compute_risk_metrics(tickers_str_st, names_str_st)
                            beta_map = dict(zip(risk_df_st['Ticker'], risk_df_st['Beta (vs STI)'].fillna(1.0))) if not risk_df_st.empty else {}
                        except Exception:
                            beta_map = {}

                    rows_crisis = []
                    for _, row in df.iterrows():
                        ticker = row['Ticker']
                        aum = row['AUM (SGD)']
                        annual_inc = row['Annual Dividend (SGD)']
                        if pd.notnull(ticker) and ticker in crisis_rets:
                            cr = crisis_rets[ticker]
                            source = "Actual data"
                        elif pd.notnull(ticker):
                            beta = float(beta_map.get(ticker, 1.0))
                            cr = scen['sti_drop'] * beta
                            source = f"Beta est. ({beta:.2f}×STI)"
                        else:
                            cr = scen['sti_drop']
                            source = "STI proxy"
                        aum_loss = aum * cr
                        stressed_aum = aum + aum_loss
                        stressed_inc = annual_inc * max(0.0, stressed_aum / aum) if aum > 0 else 0.0
                        rows_crisis.append({
                            "Security": row['Security'],
                            "Sector": row['Sector'],
                            "Current AUM (SGD)": aum,
                            "Crisis Return (%)": round(cr * 100, 1),
                            "AUM Loss (SGD)": round(aum_loss, 0),
                            "Stressed AUM (SGD)": round(stressed_aum, 0),
                            "Current Income (SGD)": round(annual_inc, 2),
                            "Stressed Income (SGD)": round(stressed_inc, 2),
                            "Data Source": source,
                        })

                    crisis_df = pd.DataFrame(rows_crisis)
                    cr_total_stressed = crisis_df['Stressed AUM (SGD)'].sum()
                    cr_aum_loss = crisis_df['AUM Loss (SGD)'].sum()
                    cr_pct_loss = (cr_aum_loss / total_aum * 100) if total_aum > 0 else 0
                    cr_stressed_inc = crisis_df['Stressed Income (SGD)'].sum()

                    ck1, ck2, ck3, ck4 = st.columns(4)
                    with ck1: kpi_card("💼", "Stressed AUM", f"S${cr_total_stressed:,.0f}")
                    with ck2: kpi_card("📉", "Estimated AUM Loss", f"S${cr_aum_loss:,.0f}")
                    with ck3: kpi_card("📊", "Portfolio Loss", f"{cr_pct_loss:.1f}%")
                    with ck4: kpi_card("💰", "Stressed Annual Income", f"S${cr_stressed_inc:,.0f}")

                    st.markdown("<br>", unsafe_allow_html=True)
                    fig_cr = go.Figure()
                    fig_cr.add_trace(go.Bar(
                        name="Current", x=["AUM (SGD)", "Annual Income (SGD)"],
                        y=[total_aum, total_inc], marker_color="#2980B9",
                        text=[f"S${total_aum:,.0f}", f"S${total_inc:,.0f}"], textposition="outside"
                    ))
                    fig_cr.add_trace(go.Bar(
                        name=f"Stressed — {scen_name}", x=["AUM (SGD)", "Annual Income (SGD)"],
                        y=[cr_total_stressed, cr_stressed_inc], marker_color="#E74C3C",
                        text=[f"S${cr_total_stressed:,.0f}", f"S${cr_stressed_inc:,.0f}"], textposition="outside"
                    ))
                    fig_cr.update_layout(barmode="group", template="plotly_white", height=360,
                                         legend=dict(orientation="h", y=1.1),
                                         margin=dict(t=30, b=20, l=20, r=20),
                                         yaxis=dict(showticklabels=False, showgrid=False))
                    st.plotly_chart(fig_cr, use_container_width=True)

                    st.markdown("#### Holdings Impact Breakdown")
                    actual_count = int((crisis_df['Data Source'] == 'Actual data').sum())
                    st.caption(f"ℹ️ {actual_count} of {len(crisis_df)} holdings used actual historical price data. Remainder estimated via beta or STI proxy.")
                    disp_cr = crisis_df[['Security', 'Sector', 'Current AUM (SGD)', 'Crisis Return (%)', 'AUM Loss (SGD)', 'Stressed AUM (SGD)', 'Data Source']].sort_values('AUM Loss (SGD)')
                    st.dataframe(
                        disp_cr.style.format({
                            'Current AUM (SGD)': 'S${:,.0f}',
                            'Crisis Return (%)': '{:+.1f}%',
                            'AUM Loss (SGD)': 'S${:,.0f}',
                            'Stressed AUM (SGD)': 'S${:,.0f}',
                        }).map(lambda v: 'color: #C0392B; font-weight: bold;' if isinstance(v, (int, float)) and v < 0 else '',
                               subset=['Crisis Return (%)', 'AUM Loss (SGD)']),
                        hide_index=True, use_container_width=True
                    )
            else:
                st.markdown("""
                <div style='text-align:center; padding:40px; background:#f4f8ff; border-radius:14px; border:1px dashed #c0d8f0; margin-top:16px;'>
                  <div style='font-size:2.5rem; margin-bottom:12px;'>📉</div>
                  <p style='font-weight:600; color:#0d2240; margin-bottom:6px;'>Crisis Scenario Simulation</p>
                  <p style='color:#6b7f9e; font-size:0.88rem; max-width:480px; margin:0 auto;'>
                    Select a scenario above and click <strong>Run Crisis Simulation</strong>.
                    Actual price data is used where available; older periods fall back to
                    beta-adjusted STI returns.
                  </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # ── MODE 2: Custom Sector Shock ──
            st.markdown("### ⚙️ Mode 2 — Custom Sector Shock")
            st.markdown('<p class="tab-subtitle" style="margin-top:-10px;">Apply a custom price shock per sector and instantly see the portfolio impact — no data download required.</p>', unsafe_allow_html=True)

            sectors_present = sorted(df['Sector'].unique())
            n_slider_cols = min(len(sectors_present), 3)
            slider_cols = st.columns(n_slider_cols)
            sector_shocks = {}
            DEFAULT_SHOCKS = {
                "REITs & Business Trusts": -20,
                "Financial Services": -15,
                "Industrials & Diversified": -20,
                "Real Estate": -20,
                "Telecommunications": -15,
                "Technology": -25,
                "Consumer Goods": -15,
                "Fixed Income": -5,
                "Equities (Unclassified)": -15,
            }
            for i, sector in enumerate(sectors_present):
                with slider_cols[i % n_slider_cols]:
                    sector_shocks[sector] = st.slider(
                        sector, min_value=-80, max_value=20,
                        value=0, format="%d%%", key=f"shock_{sector}"
                    )

            shock_rows = []
            for _, row in df.iterrows():
                shock_pct = sector_shocks.get(row['Sector'], 0) / 100.0
                aum = row['AUM (SGD)']
                annual_inc = row['Annual Dividend (SGD)']
                aum_change = aum * shock_pct
                stressed_aum = aum + aum_change
                stressed_inc = annual_inc * max(0.0, stressed_aum / aum) if aum > 0 else 0.0
                shock_rows.append({
                    "Security": row['Security'],
                    "Sector": row['Sector'],
                    "Shock (%)": shock_pct * 100,
                    "Current AUM (SGD)": aum,
                    "AUM Change (SGD)": round(aum_change, 0),
                    "Stressed AUM (SGD)": round(stressed_aum, 0),
                    "Stressed Income (SGD)": round(stressed_inc, 2),
                })

            shock_df = pd.DataFrame(shock_rows)
            cs_total_stressed = shock_df['Stressed AUM (SGD)'].sum()
            cs_aum_change = shock_df['AUM Change (SGD)'].sum()
            cs_pct = (cs_aum_change / total_aum * 100) if total_aum > 0 else 0
            cs_stressed_inc = shock_df['Stressed Income (SGD)'].sum()

            sk1, sk2, sk3, sk4 = st.columns(4)
            with sk1: kpi_card("💼", "Stressed AUM", f"S${cs_total_stressed:,.0f}")
            with sk2: kpi_card("📉", "AUM Change", f"S${cs_aum_change:,.0f}")
            with sk3: kpi_card("📊", "Portfolio Change", f"{cs_pct:.1f}%")
            with sk4: kpi_card("💰", "Stressed Annual Income", f"S${cs_stressed_inc:,.0f}")

            sector_impact = shock_df.groupby('Sector').agg(
                current_aum=('Current AUM (SGD)', 'sum'),
                stressed_aum=('Stressed AUM (SGD)', 'sum')
            ).reset_index()
            fig_cs = go.Figure()
            fig_cs.add_trace(go.Bar(
                name="Current AUM", x=sector_impact['Sector'], y=sector_impact['current_aum'],
                marker_color="#2980B9",
                text=[f"S${v:,.0f}" for v in sector_impact['current_aum']], textposition="outside"
            ))
            fig_cs.add_trace(go.Bar(
                name="Stressed AUM", x=sector_impact['Sector'], y=sector_impact['stressed_aum'],
                marker_color="#E74C3C",
                text=[f"S${v:,.0f}" for v in sector_impact['stressed_aum']], textposition="outside"
            ))
            fig_cs.update_layout(
                barmode="group", template="plotly_white", height=420,
                xaxis_tickangle=-25,
                legend=dict(orientation="h", y=1.1),
                margin=dict(t=30, b=90, l=20, r=20),
                yaxis=dict(showticklabels=False, showgrid=False)
            )
            st.plotly_chart(fig_cs, use_container_width=True)

            st.markdown("#### Holdings Impact")
            disp_cs = shock_df[['Security', 'Sector', 'Shock (%)', 'Current AUM (SGD)', 'AUM Change (SGD)', 'Stressed AUM (SGD)']].sort_values('AUM Change (SGD)')
            st.dataframe(
                disp_cs.style.format({
                    'Shock (%)': '{:+.0f}%',
                    'Current AUM (SGD)': 'S${:,.0f}',
                    'AUM Change (SGD)': 'S${:,.0f}',
                    'Stressed AUM (SGD)': 'S${:,.0f}',
                }).map(lambda v: 'color: #C0392B; font-weight: bold;' if isinstance(v, (int, float)) and v < 0 else '',
                       subset=['Shock (%)', 'AUM Change (SGD)']),
                hide_index=True, use_container_width=True
            )
            st.caption("ℹ️ Income stress is estimated proportionally to AUM change. Dividend cuts in real crises may be more severe, particularly for REITs.")

        # ══════════════════════════════════════════
        # TAB 9 — GOAL PLANNER
        # ══════════════════════════════════════════
        with t9:
            st.markdown('<p class="tab-subtitle">Model financial goals and track your portfolio\'s progress toward passive income milestones, retirement readiness, and wealth accumulation targets.</p>', unsafe_allow_html=True)

            goal_mode = st.radio(
                "Select Goal Type",
                ["💸 Passive Income", "🏖️ Retirement Readiness", "📈 Portfolio Growth"],
                horizontal=True
            )
            st.markdown("---")

            current_year = datetime.now().year
            current_yield_dec = port_yield / 100

            # Derive weighted historical return from cached risk metrics (best-effort)
            default_return_pct = round(port_yield + 2.0, 1)
            try:
                elig_gp = df[df['Ticker'].notnull()]
                if not elig_gp.empty:
                    tg = ",".join(elig_gp['Ticker'].tolist())
                    ng = "|".join(elig_gp['Security'].tolist())
                    rg = compute_risk_metrics(tg, ng)
                    if not rg.empty:
                        aum_gp = dict(zip(elig_gp['Security'], elig_gp['AUM (SGD)']))
                        tot_gp = elig_gp['AUM (SGD)'].sum()
                        wret = sum(
                            float(rg.loc[rg['Security'] == n, 'Ann. Return (%)'].iloc[0]) / 100 * aum_gp.get(n, 0)
                            for n in rg['Security'].tolist() if aum_gp.get(n, 0) > 0
                        ) / tot_gp if tot_gp > 0 else 0
                        if 0.0 < wret < 0.30:
                            default_return_pct = round(wret * 100, 1)
            except Exception:
                pass

            # ── PASSIVE INCOME ──────────────────────────────────────
            if goal_mode == "💸 Passive Income":
                st.markdown("#### 💸 Passive Income Goal")
                st.markdown("Set a target monthly dividend income and project when your portfolio reaches it — with or without reinvesting dividends.")

                current_monthly_inc = total_inc / 12

                gi_c1, gi_c2 = st.columns([1, 2])
                with gi_c1:
                    target_monthly = st.number_input(
                        "Target Monthly Income (SGD)",
                        min_value=100.0, max_value=100000.0,
                        value=float(round(max(current_monthly_inc * 1.5, current_monthly_inc + 500), -2)),
                        step=100.0
                    )
                    horizon_pi = st.slider("Projection Horizon (Years)", 5, 40, 15)
                    annual_savings_pi = st.number_input(
                        "Annual Additional Investment (SGD)",
                        min_value=0.0, max_value=1000000.0,
                        value=12000.0, step=1000.0,
                        help="New capital added each year (S$1,000/month = S$12,000/year)"
                    )
                    exp_return_pi = st.slider(
                        "Expected Annual Total Return (%)",
                        1.0, 15.0, float(min(max(default_return_pct, 3.0), 12.0)), 0.5,
                        help="Dividend yield + expected capital appreciation"
                    )

                progress_pct = min(100.0, current_monthly_inc / target_monthly * 100) if target_monthly > 0 else 0
                gap_capital = max(0.0, target_monthly * 12 / current_yield_dec - total_aum) if current_yield_dec > 0 else 0.0

                pk1, pk2, pk3, pk4 = st.columns(4)
                with pk1: kpi_card("💰", "Current Monthly Income", f"S${current_monthly_inc:,.0f}")
                with pk2: kpi_card("🎯", "Target Monthly Income", f"S${target_monthly:,.0f}")
                with pk3: kpi_card("📊", "Progress to Target", f"{progress_pct:.1f}%")
                with pk4: kpi_card("💼", "Capital Gap (at current yield)", f"S${gap_capital:,.0f}")

                bar_color_pi = "#27AE60" if progress_pct >= 80 else "#F39C12" if progress_pct >= 40 else "#E74C3C"
                st.markdown(f"""
                <div style='margin:16px 0 8px 0;'>
                  <div style='font-size:0.75rem;color:#6b7f9e;margin-bottom:4px;text-transform:uppercase;letter-spacing:1px;'>Progress to Target</div>
                  <div style='background:#e8eef7;border-radius:8px;height:14px;overflow:hidden;'>
                    <div style='width:{min(progress_pct,100):.1f}%;background:{bar_color_pi};height:14px;border-radius:8px;'></div>
                  </div>
                  <div style='font-size:0.72rem;color:#6b7f9e;margin-top:3px;'>S${current_monthly_inc:,.0f} / S${target_monthly:,.0f} per month</div>
                </div>
                """, unsafe_allow_html=True)

                exp_ret_dec_pi = exp_return_pi / 100
                proj_reinvest = project_portfolio_growth(total_aum, annual_savings_pi, exp_ret_dec_pi, current_yield_dec, horizon_pi, True)
                proj_no_reinvest = project_portfolio_growth(total_aum, annual_savings_pi, exp_ret_dec_pi, current_yield_dec, horizon_pi, False)

                proj_years_pi = [current_year + r['Year'] for r in proj_reinvest]
                inc_reinvest = [r['Monthly Income (SGD)'] for r in proj_reinvest]
                inc_no_reinvest = [r['Monthly Income (SGD)'] for r in proj_no_reinvest]

                eta_reinvest = next((current_year + r['Year'] for r in proj_reinvest if r['Monthly Income (SGD)'] >= target_monthly), None)
                eta_no_reinvest = next((current_year + r['Year'] for r in proj_no_reinvest if r['Monthly Income (SGD)'] >= target_monthly), None)

                fig_pi = go.Figure()
                fig_pi.add_trace(go.Scatter(
                    x=proj_years_pi, y=inc_reinvest,
                    name="With Dividend Reinvestment",
                    line=dict(color='#27AE60', width=2.5),
                    fill='tozeroy', fillcolor='rgba(39,174,96,0.08)'
                ))
                fig_pi.add_trace(go.Scatter(
                    x=proj_years_pi, y=inc_no_reinvest,
                    name="Dividends Paid Out (no reinvestment)",
                    line=dict(color='#F39C12', width=2, dash='dot')
                ))
                fig_pi.add_hline(
                    y=target_monthly, line_dash="dash", line_color="#E74C3C", line_width=2,
                    annotation_text=f"Target: S${target_monthly:,.0f}/month", annotation_position="top left"
                )
                fig_pi.update_layout(
                    template='plotly_white', height=380,
                    xaxis_title="Year", yaxis_title="Monthly Income (SGD)",
                    legend=dict(orientation="h", y=1.12),
                    margin=dict(t=30, b=20, l=20, r=20)
                )
                with gi_c2:
                    st.plotly_chart(fig_pi, use_container_width=True)

                eta1, eta2 = st.columns(2)
                with eta1:
                    if eta_reinvest:
                        st.success(f"**With reinvestment:** Target reached in **{eta_reinvest}** ({eta_reinvest - current_year} years)")
                    else:
                        st.warning(f"**With reinvestment:** Reaches S${inc_reinvest[-1]:,.0f}/month by {proj_years_pi[-1]}. Extend horizon or increase savings.")
                with eta2:
                    if eta_no_reinvest:
                        st.info(f"**Without reinvestment:** Target reached in **{eta_no_reinvest}** ({eta_no_reinvest - current_year} years)")
                    else:
                        st.warning(f"**Without reinvestment:** Reaches S${inc_no_reinvest[-1]:,.0f}/month by {proj_years_pi[-1]}.")

                st.markdown("#### Year-by-Year Projection (With Reinvestment)")
                proj_df_pi = pd.DataFrame(proj_reinvest)
                proj_df_pi['Calendar Year'] = proj_years_pi
                proj_df_pi_disp = proj_df_pi[['Calendar Year', 'AUM (SGD)', 'Annual Income (SGD)', 'Monthly Income (SGD)']].rename(
                    columns={'AUM (SGD)': 'Portfolio AUM', 'Annual Income (SGD)': 'Annual Dividends', 'Monthly Income (SGD)': 'Monthly Dividends'}
                )
                st.dataframe(
                    proj_df_pi_disp.style.format({
                        'Portfolio AUM': 'S${:,.0f}',
                        'Annual Dividends': 'S${:,.0f}',
                        'Monthly Dividends': 'S${:,.0f}',
                    }).background_gradient(subset=['Monthly Dividends'], cmap='Greens', vmin=0),
                    hide_index=True, use_container_width=True
                )
                st.caption(f"ℹ️ Assumes {exp_return_pi:.1f}% total annual return and S${annual_savings_pi:,.0f}/year new capital. Portfolio yield held constant at {port_yield:.2f}%.")

            # ── RETIREMENT READINESS ─────────────────────────────────
            elif goal_mode == "🏖️ Retirement Readiness":
                st.markdown("#### 🏖️ Retirement Readiness")
                st.markdown("Model the AUM required to fund your retirement income target and check whether your portfolio is on track.")

                rr_c1, rr_c2, rr_c3 = st.columns(3)
                with rr_c1:
                    current_age = st.slider("Current Age", 25, 74, 50)
                    retirement_age = st.slider("Target Retirement Age", current_age + 1, 85, min(current_age + 15, 65))
                with rr_c2:
                    target_ret_monthly = st.number_input(
                        "Target Monthly Income at Retirement (SGD)",
                        min_value=500.0, max_value=50000.0, value=3000.0, step=500.0
                    )
                    withdrawal_yield = st.slider(
                        "Withdrawal Yield at Retirement (%)", 1.0, 10.0,
                        float(round(min(max(port_yield, 2.0), 8.0), 1)), 0.5,
                        help="Expected dividend yield of your portfolio at retirement"
                    )
                with rr_c3:
                    exp_return_rr = st.slider(
                        "Expected Annual Total Return (%)", 1.0, 15.0,
                        float(min(max(default_return_pct, 3.0), 12.0)), 0.5
                    )
                    annual_savings_rr = st.number_input(
                        "Annual Additional Investment (SGD)",
                        min_value=0.0, max_value=1000000.0, value=12000.0, step=1000.0
                    )

                years_to_retire = retirement_age - current_age
                req_aum = target_ret_monthly * 12 / (withdrawal_yield / 100) if withdrawal_yield > 0 else 0.0
                proj_rr = project_portfolio_growth(total_aum, annual_savings_rr, exp_return_rr / 100, current_yield_dec, years_to_retire, True)
                projected_aum_retirement = float(proj_rr[-1]['AUM (SGD)']) if proj_rr else total_aum
                aum_gap_rr = req_aum - projected_aum_retirement
                readiness_pct = min(100.0, projected_aum_retirement / req_aum * 100) if req_aum > 0 else 100.0

                r_rr = exp_return_rr / 100
                n_rr = years_to_retire
                if aum_gap_rr > 0 and n_rr > 0 and r_rr > 0:
                    extra_savings_rr = aum_gap_rr * r_rr / ((1 + r_rr) ** n_rr - 1)
                elif aum_gap_rr > 0 and n_rr > 0:
                    extra_savings_rr = aum_gap_rr / n_rr
                else:
                    extra_savings_rr = 0.0

                rk1, rk2, rk3, rk4 = st.columns(4)
                with rk1: kpi_card("🎯", "AUM Needed at Retirement", f"S${req_aum:,.0f}")
                with rk2: kpi_card("📈", f"Projected AUM (in {n_rr}Y)", f"S${projected_aum_retirement:,.0f}")
                with rk3: kpi_card("📊", "Retirement Readiness", f"{readiness_pct:.1f}%")
                with rk4:
                    if aum_gap_rr > 0:
                        kpi_card("⚠️", "AUM Gap", f"S${aum_gap_rr:,.0f}")
                    else:
                        kpi_card("✅", "AUM Surplus", f"S${abs(aum_gap_rr):,.0f}")

                bar_color_rr = "#27AE60" if readiness_pct >= 80 else "#F39C12" if readiness_pct >= 50 else "#E74C3C"
                st.markdown(f"""
                <div style='margin:16px 0 8px 0;'>
                  <div style='font-size:0.75rem;color:#6b7f9e;margin-bottom:4px;text-transform:uppercase;letter-spacing:1px;'>Retirement Readiness</div>
                  <div style='background:#e8eef7;border-radius:8px;height:14px;overflow:hidden;'>
                    <div style='width:{min(readiness_pct,100):.1f}%;background:{bar_color_rr};height:14px;border-radius:8px;'></div>
                  </div>
                  <div style='font-size:0.72rem;color:#6b7f9e;margin-top:3px;'>Projected S${projected_aum_retirement:,.0f} vs required S${req_aum:,.0f} at age {retirement_age}</div>
                </div>
                """, unsafe_allow_html=True)

                if aum_gap_rr > 0:
                    st.warning(f"**Gap of S${aum_gap_rr:,.0f}** remains. An additional **S${extra_savings_rr:,.0f}/year** (S${extra_savings_rr/12:,.0f}/month) on top of current savings would close the gap by age {retirement_age}, at {exp_return_rr:.1f}% annual return.")
                else:
                    st.success(f"Portfolio is on track — projected to exceed the retirement target by **S${abs(aum_gap_rr):,.0f}**. Consider increasing your income target or retiring earlier.")

                proj_years_rr = [current_year + r['Year'] for r in proj_rr]
                fig_rr = go.Figure()
                fig_rr.add_trace(go.Scatter(
                    x=proj_years_rr, y=[r['AUM (SGD)'] for r in proj_rr],
                    name="Projected Portfolio AUM",
                    line=dict(color='#2980B9', width=2.5),
                    fill='tozeroy', fillcolor='rgba(41,128,185,0.08)'
                ))
                fig_rr.add_hline(
                    y=req_aum, line_dash="dash", line_color="#E74C3C", line_width=2,
                    annotation_text=f"Required AUM: S${req_aum:,.0f}", annotation_position="top left"
                )
                fig_rr.add_vline(
                    x=current_year + n_rr, line_dash="dot", line_color="#F39C12", line_width=2,
                    annotation_text=f"Retirement {current_year + n_rr}", annotation_position="top right"
                )
                fig_rr.update_layout(
                    template='plotly_white', height=380,
                    xaxis_title="Year", yaxis_title="Portfolio AUM (SGD)",
                    legend=dict(orientation="h", y=1.12),
                    margin=dict(t=30, b=20, l=20, r=20)
                )
                st.plotly_chart(fig_rr, use_container_width=True)
                st.caption(f"ℹ️ Assumes {exp_return_rr:.1f}% annual total return with full dividend reinvestment, S${annual_savings_rr:,.0f}/year new investment. Required AUM = S${target_ret_monthly:,.0f}/month × 12 ÷ {withdrawal_yield:.1f}% withdrawal yield.")

                st.markdown("#### Retirement Scenario Comparison")
                scen_rows = []
                for ret_scenario, label in [(exp_return_rr - 2, "Pessimistic"), (exp_return_rr, "Base"), (exp_return_rr + 2, "Optimistic")]:
                    if ret_scenario <= 0:
                        continue
                    p_s = project_portfolio_growth(total_aum, annual_savings_rr, ret_scenario / 100, current_yield_dec, n_rr, True)
                    aum_s = float(p_s[-1]['AUM (SGD)'])
                    inc_s = aum_s * (withdrawal_yield / 100) / 12
                    gap_s = req_aum - aum_s
                    scen_rows.append({
                        "Scenario": label,
                        "Annual Return": f"{ret_scenario:.1f}%",
                        "AUM at Retirement (SGD)": round(aum_s, 0),
                        "Monthly Income (SGD)": round(inc_s, 0),
                        "Gap to Target (SGD)": round(gap_s, 0),
                    })
                if scen_rows:
                    scen_df_rr = pd.DataFrame(scen_rows)
                    st.dataframe(
                        scen_df_rr.style.format({
                            'AUM at Retirement (SGD)': 'S${:,.0f}',
                            'Monthly Income (SGD)': 'S${:,.0f}',
                            'Gap to Target (SGD)': 'S${:+,.0f}',
                        }).map(lambda v: 'color: #27AE60; font-weight:bold;' if isinstance(v, (int, float)) and v <= 0 else
                               ('color: #C0392B; font-weight:bold;' if isinstance(v, (int, float)) and v > 0 else ''),
                               subset=['Gap to Target (SGD)']),
                        hide_index=True, use_container_width=True
                    )

            # ── PORTFOLIO GROWTH TARGET ──────────────────────────────
            else:
                st.markdown("#### 📈 Portfolio Growth Target")
                st.markdown("Set a target AUM and project when you'll reach it under bear, base, and bull return scenarios.")

                pg_c1, pg_c2 = st.columns([1, 2])
                with pg_c1:
                    target_aum_pg = st.number_input(
                        "Target Portfolio AUM (SGD)",
                        min_value=10000.0, max_value=100000000.0,
                        value=float(round(total_aum * 2, -4)), step=50000.0
                    )
                    annual_savings_pg = st.number_input(
                        "Annual Additional Investment (SGD)",
                        min_value=0.0, max_value=1000000.0, value=12000.0, step=1000.0
                    )
                    exp_return_pg = st.slider(
                        "Base Expected Annual Return (%)", 1.0, 15.0,
                        float(min(max(default_return_pct, 3.0), 12.0)), 0.5
                    )
                    reinvest_pg = st.checkbox("Reinvest Dividends", value=True)

                horizon_pg = 40
                bear_ret = max(0.5, exp_return_pg - 3.0) / 100
                base_ret = exp_return_pg / 100
                bull_ret = (exp_return_pg + 3.0) / 100

                proj_bear = project_portfolio_growth(total_aum, annual_savings_pg, bear_ret, current_yield_dec, horizon_pg, reinvest_pg)
                proj_base = project_portfolio_growth(total_aum, annual_savings_pg, base_ret, current_yield_dec, horizon_pg, reinvest_pg)
                proj_bull = project_portfolio_growth(total_aum, annual_savings_pg, bull_ret, current_yield_dec, horizon_pg, reinvest_pg)

                proj_years_pg = [current_year + r['Year'] for r in proj_base]
                eta_bear = next((current_year + r['Year'] for r in proj_bear if r['AUM (SGD)'] >= target_aum_pg), None)
                eta_base = next((current_year + r['Year'] for r in proj_base if r['AUM (SGD)'] >= target_aum_pg), None)
                eta_bull = next((current_year + r['Year'] for r in proj_bull if r['AUM (SGD)'] >= target_aum_pg), None)
                gap_pg = max(0.0, target_aum_pg - total_aum)

                pgk1, pgk2, pgk3, pgk4 = st.columns(4)
                with pgk1: kpi_card("🎯", "Target AUM", f"S${target_aum_pg:,.0f}")
                with pgk2: kpi_card("💼", "Current AUM", f"S${total_aum:,.0f}")
                with pgk3: kpi_card("📉", "AUM Gap", f"S${gap_pg:,.0f}")
                with pgk4:
                    kpi_card("📅", "Base Case ETA", str(eta_base) if eta_base else ">40Y")

                fig_pg = go.Figure()
                fig_pg.add_trace(go.Scatter(
                    x=proj_years_pg, y=[r['AUM (SGD)'] for r in proj_bear],
                    name=f"Bear ({exp_return_pg - 3:.1f}%)",
                    line=dict(color='#E74C3C', width=1.5, dash='dot')
                ))
                fig_pg.add_trace(go.Scatter(
                    x=proj_years_pg, y=[r['AUM (SGD)'] for r in proj_base],
                    name=f"Base ({exp_return_pg:.1f}%)",
                    line=dict(color='#2980B9', width=2.5),
                    fill='tonexty', fillcolor='rgba(41,128,185,0.07)'
                ))
                fig_pg.add_trace(go.Scatter(
                    x=proj_years_pg, y=[r['AUM (SGD)'] for r in proj_bull],
                    name=f"Bull ({exp_return_pg + 3:.1f}%)",
                    line=dict(color='#27AE60', width=1.5, dash='dot'),
                    fill='tonexty', fillcolor='rgba(39,174,96,0.06)'
                ))
                fig_pg.add_hline(
                    y=target_aum_pg, line_dash="dash", line_color="#F39C12", line_width=2,
                    annotation_text=f"Target: S${target_aum_pg:,.0f}", annotation_position="top left"
                )
                fig_pg.update_layout(
                    template='plotly_white', height=420,
                    xaxis_title="Year", yaxis_title="Portfolio AUM (SGD)",
                    legend=dict(orientation="h", y=1.12),
                    margin=dict(t=30, b=20, l=20, r=20)
                )
                with pg_c2:
                    st.plotly_chart(fig_pg, use_container_width=True)

                with pg_c1:
                    st.markdown("**Scenario ETAs**")
                    st.write(f"🐻 Bear ({exp_return_pg-3:.1f}%): **{eta_bear if eta_bear else '>40Y'}**" + (f" ({eta_bear - current_year}Y)" if eta_bear else ""))
                    st.write(f"📊 Base ({exp_return_pg:.1f}%): **{eta_base if eta_base else '>40Y'}**" + (f" ({eta_base - current_year}Y)" if eta_base else ""))
                    st.write(f"🐂 Bull ({exp_return_pg+3:.1f}%): **{eta_bull if eta_bull else '>40Y'}**" + (f" ({eta_bull - current_year}Y)" if eta_bull else ""))

                st.markdown("#### Return Sensitivity")
                sens_rows = []
                for ret_pct in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]:
                    ps = project_portfolio_growth(total_aum, annual_savings_pg, ret_pct / 100, current_yield_dec, 40, reinvest_pg)
                    eta_s = next((r['Year'] for r in ps if r['AUM (SGD)'] >= target_aum_pg), None)
                    sens_rows.append({
                        "Annual Return": f"{ret_pct:.0f}%",
                        "AUM in 10Y": ps[min(10, len(ps)-1)]['AUM (SGD)'],
                        "AUM in 20Y": ps[min(20, len(ps)-1)]['AUM (SGD)'],
                        "AUM in 30Y": ps[min(30, len(ps)-1)]['AUM (SGD)'],
                        "Years to Target": f"{eta_s}Y" if eta_s else ">40Y",
                    })
                sens_df = pd.DataFrame(sens_rows)
                st.dataframe(
                    sens_df.style.format({
                        'AUM in 10Y': 'S${:,.0f}',
                        'AUM in 20Y': 'S${:,.0f}',
                        'AUM in 30Y': 'S${:,.0f}',
                    }),
                    hide_index=True, use_container_width=True
                )
                st.caption(f"ℹ️ All scenarios assume S${annual_savings_pg:,.0f}/year new investment, {'with' if reinvest_pg else 'without'} dividend reinvestment. Portfolio yield held constant at {port_yield:.2f}%.")

# ══════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════
else:
    st.markdown("""
    <div style='text-align:center; padding: 40px 0 10px 0;'>
      <div style='font-size:3.5rem;'>🏦</div>
      <h1 style='color:#0d2240; font-size:2rem; font-weight:700; margin:12px 0 6px 0;'>CLP Wealth Center</h1>
      <p style='color:#6b7f9e; font-size:1rem; max-width:540px; margin:0 auto;'>
        Your personal portfolio intelligence platform. Upload a CDP statement PDF to unlock
        nine analytical modules — from sector analysis and income tracking to risk metrics,
        dividend forecasting, portfolio optimisation, crisis stress testing, and goal-based planning.
      </p>
    </div>
    <div class="feat-grid" style='margin-top:36px;'>
      <div class="feat-card"><div class="feat-icon">📊</div><div class="feat-title">Asset Discovery</div><div class="feat-desc">Treemap, sunburst and scatter charts reveal capital and income distribution across sectors.</div></div>
      <div class="feat-card"><div class="feat-icon">📈</div><div class="feat-title">Market Benchmark</div><div class="feat-desc">Compare any holding's total return against the STI index over 1–20 years.</div></div>
      <div class="feat-card"><div class="feat-icon">📂</div><div class="feat-title">Verification Hub</div><div class="feat-desc">Drill into each sector, review DPS and yield, and export data to CSV.</div></div>
      <div class="feat-card"><div class="feat-icon">📜</div><div class="feat-title">Strategic Advisory</div><div class="feat-desc">Health gauge, creator/destroyer audit, and rebalance simulator.</div></div>
      <div class="feat-card"><div class="feat-icon">⚠️</div><div class="feat-title">Risk Analytics</div><div class="feat-desc">Volatility, Sharpe ratio, beta, VaR, max drawdown, and correlation heatmap per holding.</div></div>
      <div class="feat-card"><div class="feat-icon">📅</div><div class="feat-title">Dividend Calendar</div><div class="feat-desc">12-month income projection by holding, stacked by month with a gap analysis.</div></div>
      <div class="feat-card"><div class="feat-icon">🎯</div><div class="feat-title">Portfolio Optimisation</div><div class="feat-desc">Efficient Frontier via Monte Carlo — Max Sharpe and Min Volatility weight suggestions.</div></div>
      <div class="feat-card"><div class="feat-icon">🔥</div><div class="feat-title">Stress Test</div><div class="feat-desc">Simulate portfolio impact under GFC, COVID, and rate-hike crises, or apply custom sector shocks.</div></div>
      <div class="feat-card"><div class="feat-icon">🏆</div><div class="feat-title">Goal Planner</div><div class="feat-desc">Model passive income, retirement readiness, and portfolio growth targets with projection charts.</div></div>
    </div>
    <div style='text-align:center; margin-top:40px; padding:24px; background:#f4f8ff; border-radius:14px; border:1px dashed #c0d8f0;'>
      <div style='font-size:1.4rem; margin-bottom:8px;'>⬆️</div>
      <p style='color:#0d2240; font-weight:600; margin:0 0 4px 0;'>Get started</p>
      <p style='color:#6b7f9e; font-size:0.88rem; margin:0;'>Use the <strong>Upload CDP Portfolio Statement</strong> button in the sidebar to begin.</p>
    </div>
    """, unsafe_allow_html=True)
