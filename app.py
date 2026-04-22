import streamlit as st
import pandas as pd
import pdfplumber
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import re
import numpy as np
import requests
from datetime import datetime, timedelta

# --- 1. SYSTEM CONFIG ---
st.set_page_config(layout="wide", page_title="CDP Wealth Center", page_icon="🏦")

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

/* Sidebar collapse/expand button */
[data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    background: transparent !important;
    border: none !important;
}
[data-testid="stSidebarCollapseButton"] svg {
    stroke: #8fa8cc !important;
    fill: none !important;
}
[data-testid="stSidebarCollapseButton"]:hover svg {
    stroke: #f0b429 !important;
}

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
button[data-baseweb="tab"]:focus-visible {
    outline: 2px solid #f0b429 !important;
    outline-offset: -2px !important;
    background: rgba(240, 180, 41, 0.12) !important;
    color: #0d2240 !important;
    border-radius: 8px 8px 0 0 !important;
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
# Edit intel_data.py to add/update stocks, DPS rates, and aliases.
from intel_data import _DPS_AS_OF, DPS_NOTES, MASTER_INTEL, ALIAS_MAP

# Case-insensitive lookup indexes for resolve_intel — built once at import time.
_MASTER_INTEL_UP = {k.upper(): v for k, v in MASTER_INTEL.items()}
_ALIAS_MAP_UP = {k.upper(): v for k, v in ALIAS_MAP.items()}

def _si(ticker):
    """Append .SI exchange suffix to bare SGX codes for Yahoo Finance."""
    t = str(ticker).strip() if ticker else ""
    if t and '.' not in t and not t.startswith('^'):
        return t + '.SI'
    return t

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
    if name_up in _MASTER_INTEL_UP:
        return _MASTER_INTEL_UP[name_up], name_up, True
    for clean_name, aliases in _ALIAS_MAP_UP.items():
        if any(alias in name_up for alias in aliases):
            intel = _MASTER_INTEL_UP.get(clean_name)
            if intel:
                return intel, clean_name, True
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

# --- FinBERT SENTIMENT HELPERS ---

_FINBERT_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
_SENTIMENT_COLOUR = {"positive": "#27AE60", "neutral": "#7F8C8D", "negative": "#E74C3C"}
_SENTIMENT_LABEL  = {"positive": "Bullish", "neutral": "Neutral", "negative": "Bearish"}
_SENTIMENT_ICON   = {"positive": "📈", "neutral": "➡️", "negative": "📉"}

def _call_finbert(texts: list[str], hf_token: str) -> list[dict] | None:
    """POST a batch of texts to FinBERT; return list of {label, score} dicts or None on error."""
    if not hf_token or not texts:
        return None
    try:
        resp = requests.post(
            _FINBERT_URL,
            headers={"Authorization": f"Bearer {hf_token}"},
            json={"inputs": texts},
            timeout=30,
        )
        if resp.status_code == 503:
            # Model is loading — surface a friendly message upstream
            return "loading"
        if resp.status_code != 200:
            return None
        raw = resp.json()
        # raw is list[list[{label,score}]] — pick the top label per text
        results = []
        for item in raw:
            if isinstance(item, list):
                top = max(item, key=lambda x: x["score"])
                results.append({"label": top["label"].lower(), "score": top["score"]})
            else:
                results.append({"label": "neutral", "score": 1.0})
        return results
    except Exception:
        return None

def _fetch_news_headlines(ticker_si: str, max_headlines: int = 5) -> list[str]:
    """Return up to max_headlines news titles for a .SI ticker via yfinance."""
    try:
        news = yf.Ticker(ticker_si).news or []
        return [item.get("title", "") for item in news[:max_headlines] if item.get("title")]
    except Exception:
        return []

def _aggregate_sentiment(scores: list[dict]) -> dict:
    """Collapse a list of per-headline {label,score} into a single summary dict."""
    if not scores:
        return {"label": "neutral", "net": 0.0, "pos": 0, "neu": 0, "neg": 0, "count": 0}
    pos = sum(s["score"] for s in scores if s["label"] == "positive")
    neg = sum(s["score"] for s in scores if s["label"] == "negative")
    neu = sum(s["score"] for s in scores if s["label"] == "neutral")
    n   = len(scores)
    net = (pos - neg) / n
    if net > 0.15:
        label = "positive"
    elif net < -0.15:
        label = "negative"
    else:
        label = "neutral"
    return {
        "label": label,
        "net":   round(net, 3),
        "pos":   sum(1 for s in scores if s["label"] == "positive"),
        "neu":   sum(1 for s in scores if s["label"] == "neutral"),
        "neg":   sum(1 for s in scores if s["label"] == "negative"),
        "count": n,
    }

@st.cache_data(ttl=3600)
def fetch_prices_batch(tickers_str, period="2y"):
    """Batch-download closing prices; returns DataFrame with tickers as columns."""
    raw_tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    si_tickers = [_si(t) for t in raw_tickers]
    si_to_raw = {s: r for s, r in zip(si_tickers, raw_tickers)}
    result = {}
    if not si_tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(si_tickers, period=period, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            close_df = raw['Close']
            for si, bare in si_to_raw.items():
                if si in close_df.columns:
                    s = close_df[si].dropna()
                    if not s.empty:
                        result[bare] = s
        else:
            c = raw['Close'] if 'Close' in raw.columns else raw.iloc[:, 0]
            if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
            if not c.dropna().empty:
                result[raw_tickers[0]] = c
    except Exception:
        pass
    for si, bare in [(s, r) for s, r in si_to_raw.items() if r not in result]:
        try:
            h = yf.download(si, period=period, auto_adjust=True, progress=False)
            c = h['Close'].iloc[:, 0] if isinstance(h['Close'], pd.DataFrame) else h['Close']
            if not c.dropna().empty:
                result[bare] = c
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
            divs = yf.Ticker(_si(ticker)).dividends
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
            h = yf.download(_si(ticker), period="5y", auto_adjust=True, progress=False)
            if not h.empty:
                c = h['Close'].iloc[:, 0] if isinstance(h['Close'], pd.DataFrame) else h['Close']
                m[name] = float(c.iloc[-1]) / float(c.iloc[0]) - 1.0
        except Exception:
            continue
    return m

@st.cache_data(ttl=3600)
def load_benchmark(ticker, years):
    start = datetime.now() - timedelta(days=years * 365)
    h_a = yf.download(_si(ticker), start=start, end=datetime.now(), auto_adjust=True, progress=False)
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
    "Dotcom Crash 2000–02": {
        "start": "2000-03-10", "end": "2002-10-09",
        "desc": "Dot-com bubble burst — tech collapse and prolonged bear market",
        "sti_drop": -0.542,
    },
    "European PIGS Crisis 2010–12": {
        "start": "2010-04-23", "end": "2012-07-26",
        "desc": "Eurozone sovereign debt crisis — Greece, Portugal, Ireland, Spain, Italy",
        "sti_drop": -0.197,
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
    raw_tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    si_tickers = [_si(t) for t in raw_tickers]
    si_to_raw = {s: r for s, r in zip(si_tickers, raw_tickers)}
    result = {}
    if not si_tickers:
        return result
    try:
        raw = yf.download(si_tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            close_df = raw['Close']
            for si, bare in si_to_raw.items():
                if si in close_df.columns:
                    s = close_df[si].dropna()
                    if len(s) >= 2:
                        result[bare] = float(s.iloc[-1]) / float(s.iloc[0]) - 1.0
        else:
            c = raw['Close'] if 'Close' in raw.columns else raw.iloc[:, 0]
            if isinstance(c, pd.DataFrame):
                c = c.iloc[:, 0]
            s = c.dropna()
            if len(s) >= 2:
                result[raw_tickers[0]] = float(s.iloc[-1]) / float(s.iloc[0]) - 1.0
    except Exception:
        pass
    for si, bare in [(s, r) for s, r in si_to_raw.items() if r not in result]:
        try:
            h = yf.download(si, start=start_date, end=end_date, auto_adjust=True, progress=False)
            c = h['Close'].iloc[:, 0] if isinstance(h['Close'], pd.DataFrame) else h['Close']
            s = c.dropna()
            if len(s) >= 2:
                result[bare] = float(s.iloc[-1]) / float(s.iloc[0]) - 1.0
        except Exception:
            continue
    return result

@st.cache_data(ttl=3600)
def fetch_annual_dps(ticker):
    """Sum last 12 months of dividends from Yahoo Finance for an unknown stock."""
    if not ticker or not ticker.strip():
        return 0.0
    try:
        divs = yf.Ticker(_si(ticker.strip())).dividends
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
  <div style='font-size:1.05rem; font-weight:700; color:#f0b429; letter-spacing:1.5px; margin-top:6px;'>CDP WEALTH CENTER</div>
  <div style='font-size:0.72rem; color:#8fa8cc; margin-top:4px; letter-spacing:0.5px;'>Portfolio Intelligence Platform</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("**Upload Portfolio Statement**")
uploaded_file = st.sidebar.file_uploader("PDF only", type="pdf", label_visibility="collapsed")
st.sidebar.markdown("---")
with st.sidebar.expander("🤖 AI Sentiment (FinBERT)", expanded=False):
    st.markdown(
        "<small>Enter your free <a href='https://huggingface.co/settings/tokens' target='_blank'>"
        "HuggingFace token</a> to enable the Sentiment Radar in the Advisory tab.</small>",
        unsafe_allow_html=True,
    )
    hf_token_input = st.text_input(
        "HuggingFace API Token",
        type="password",
        placeholder="hf_••••••••••••••••••••",
        label_visibility="collapsed",
    )
    if hf_token_input:
        st.session_state["hf_token"] = hf_token_input
        st.success("Token saved for this session.")
    elif "hf_token" not in st.session_state:
        st.session_state["hf_token"] = ""

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:0.72rem;color:#8fa8cc;line-height:1.55;padding:4px 2px 8px 2px;'>"
    "<strong style='color:#a0b8d0;'>⚠️ Disclaimer</strong><br>"
    "This app is provided for <strong>informational and educational purposes only</strong>. "
    "It does not constitute financial advice, investment advice, or any form of recommendation to buy, "
    "sell, or hold any security. All data, projections, and analytics are estimates based on "
    "historical information and user-supplied inputs — they may be incomplete or inaccurate. "
    "Past performance is not indicative of future results. "
    "You are solely responsible for your own investment decisions. "
    "The developer accepts no liability for any financial loss arising from use of this app. "
    "Always consult a licensed financial adviser before making investment decisions."
    "</div>",
    unsafe_allow_html=True
)

# --- 6. TABS ---
t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs([
    "📊 Discovery", "📈 Benchmark", "📂 Holdings",
    "📜 Advisory", "⚠️ Risk", "📅 Dividends",
    "🎯 Optimise", "🔥 Stress Test", "🏆 Goals"
])

# Make every tab button a TAB-key stop.
# BaseWeb's roving-tabindex pattern sets tabindex="-1" on non-selected tabs,
# so only the active tab is reachable by keyboard. This patch overrides that.
# The observer watches ONLY the tab-list element (not the whole body) to avoid
# a mutation cascade during heavy Streamlit re-renders such as PDF loading.
st.iframe("""
<script>
(function () {
    var doc = window.parent.document;
    var pending = false;

    function patch() {
        if (pending) return;
        pending = true;
        requestAnimationFrame(function () {
            doc.querySelectorAll('button[data-baseweb="tab"]').forEach(function (btn) {
                // Only write when value actually needs to change — prevents re-triggering
                if (btn.getAttribute('tabindex') !== '0') {
                    btn.setAttribute('tabindex', '0');
                }
            });
            pending = false;
        });
    }

    function attach() {
        var tabList = doc.querySelector('[data-baseweb="tab-list"]');
        if (tabList) {
            patch();
            // Watch only the tab-list for tabindex changes — never the full body
            new MutationObserver(patch).observe(tabList, {
                subtree: true,
                attributes: true,
                attributeFilter: ['tabindex']
            });
        } else {
            setTimeout(attach, 200);
        }
    }

    attach();
})();
</script>
""", height=1)

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

        # Fallback: auto-fetch DPS from yfinance for MASTER_INTEL stocks where Rate = 0
        yf_fetched_dps = []
        if 'Known' in df.columns:
            zero_known = df[(df['Known'] == True) & (df['DPS'] == 0) & (df['Ticker'].notnull())]
            for idx in zero_known.index:
                fetched = fetch_annual_dps(df.at[idx, 'Ticker'])
                if fetched > 0:
                    df.at[idx, 'DPS'] = fetched
                    yf_fetched_dps.append(df.at[idx, 'Security'])

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
        # Note stocks whose DPS was auto-fetched from yfinance
        if yf_fetched_dps:
            note_items += (
                f"<li style='margin-bottom:4px;color:#1A6BBF;'><strong>Live DPS (yfinance):</strong> "
                f"{', '.join(yf_fetched_dps)} — Rate was 0 in MASTER_INTEL; trailing 12-month dividend used instead. Update Rate to lock in a specific value.</li>"
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
            _scatter_df = df[df['Sector'] != 'Fixed Income']
            fig_scatter = px.scatter(_scatter_df, x="P/B Ratio", y="Dividend Yield (%)",
                                     size="AUM (SGD)", color="Sector", hover_name="Security", template="plotly_dark", height=500,
                                     size_max=40)
            fig_scatter.update_layout(
                margin=dict(t=20, l=10, r=10, b=10),
                yaxis=dict(range=[0, 15], title="Dividend Yield (%)"),
            )
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
                audit_df = pd.DataFrame(
                    audit_rows,
                    columns=["Security", "Sector", "DPS Used (SGD)", "Quantity",
                             "Annual Income (SGD)", "% of Total Income", "Flag", "Note"],
                )

                def _flag_style(val):
                    if isinstance(val, str) and "⚠️" in val:
                        return 'background-color:rgba(243,156,18,0.18);color:#7D6608;font-weight:600;'
                    if isinstance(val, str) and "✅" in val:
                        return 'background-color:rgba(39,174,96,0.12);color:#1E8449;'
                    return ''

                if audit_df.empty:
                    st.info("No holdings match the current sector filter.")
                else:
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
            # ── Weight-adjusted health score (4 components) ──────────────
            _tot_elig_aum = eligible_df['AUM (SGD)'].sum() if not eligible_df.empty else 1.0
            _n_elig = len(p_map)

            # C1 Capital Quality (40 pts): penalises by AUM weight of 5Y losers
            _loser_df = eligible_df[eligible_df['Security'].isin(losers)]
            _loser_wt = _loser_df['AUM (SGD)'].sum() / _tot_elig_aum if _tot_elig_aum > 0 else 0
            _c1 = 40.0 * (1.0 - _loser_wt)

            # C2 Income Coverage (30 pts): AUM-weighted share of portfolio paying dividends
            _c2 = 30.0 * (df[df['DPS'] > 0]['AUM (SGD)'].sum() / total_aum if total_aum > 0 else 0)

            # C3 Concentration (20 pts): penalises if any single holding exceeds 20% of AUM
            _top_wt = (df['AUM (SGD)'] / total_aum).max() if total_aum > 0 else 0
            _c3 = 20.0 * max(0.0, 1.0 - max(0.0, _top_wt - 0.20) / 0.40)

            # C4 Winner Breadth (10 pts): share of eligible holdings with positive 5Y return
            _c4 = 10.0 * (len(winners) / _n_elig) if _n_elig > 0 else 0

            health = round(_c1 + _c2 + _c3 + _c4)

            h_col1, h_col2 = st.columns([1, 1])
            with h_col1:
                st.plotly_chart(health_gauge(health), use_container_width=True)
            with h_col2:
                st.markdown("<br>", unsafe_allow_html=True)

                def _score_bar(score, max_score, color):
                    pct = int(score / max_score * 100)
                    return (
                        f"<div style='background:#dde6f0;border-radius:4px;height:7px;width:100%;'>"
                        f"<div style='background:{color};width:{pct}%;height:7px;border-radius:4px;'></div></div>"
                    )

                _components = [
                    ("Capital Quality",  _c1, 40, "#1E8449" if _c1 >= 28 else "#B7770D" if _c1 >= 16 else "#C0392B"),
                    ("Income Coverage",  _c2, 30, "#1E8449" if _c2 >= 21 else "#B7770D" if _c2 >= 12 else "#C0392B"),
                    ("Concentration",    _c3, 20, "#1E8449" if _c3 >= 14 else "#B7770D" if _c3 >= 8  else "#C0392B"),
                    ("Winner Breadth",   _c4, 10, "#1E8449" if _c4 >= 7  else "#B7770D" if _c4 >= 4  else "#C0392B"),
                ]
                _tbl = (
                    "<table style='width:100%;border-collapse:collapse;font-size:0.81rem;'>"
                    "<tr>"
                    "<th style='text-align:left;color:#4a6a8a;padding:2px 6px;font-weight:600;'>Component</th>"
                    "<th style='text-align:right;color:#4a6a8a;padding:2px 6px;'>Score</th>"
                    "<th style='text-align:right;color:#4a6a8a;padding:2px 6px;'>Max</th>"
                    "<th style='padding:2px 6px;width:38%;'></th>"
                    "</tr>"
                )
                for _lbl, _s, _mx, _col in _components:
                    _tbl += (
                        f"<tr>"
                        f"<td style='padding:5px 6px;color:#1a2a3a;font-weight:600;'>{_lbl}</td>"
                        f"<td style='text-align:right;padding:5px 6px;color:{_col};font-weight:700;'>{_s:.1f}</td>"
                        f"<td style='text-align:right;padding:5px 6px;color:#4a6a8a;'>{_mx}</td>"
                        f"<td style='padding:5px 10px;vertical-align:middle;'>{_score_bar(_s, _mx, _col)}</td>"
                        f"</tr>"
                    )
                _tbl += "</table>"
                st.markdown(_tbl, unsafe_allow_html=True)
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

                # Strength commentary
                top_divs = df.nlargest(2, 'Annual Dividend (SGD)')['Security'].tolist()
                _tot_inc = df['Annual Dividend (SGD)'].sum()
                if top_divs and _tot_inc > 0:
                    _top2_inc_pct = df.nlargest(2, 'Annual Dividend (SGD)')['Annual Dividend (SGD)'].sum() / _tot_inc * 100
                    anchor_text = f"**{', '.join(top_divs)}**"
                    st.markdown(f"**Strength:** Income anchored by {anchor_text}, contributing **{_top2_inc_pct:.0f}%** of total dividends.")
                else:
                    st.markdown("**Strength:** Portfolio spans multiple income-generating holdings.")

                # Priority commentary — weighted, names the biggest drags
                if losers:
                    _loser_weights = {
                        l: eligible_df.loc[eligible_df['Security'] == l, 'AUM (SGD)'].sum() / _tot_elig_aum * 100
                        for l in losers
                    }
                    _top_drags = sorted(_loser_weights.items(), key=lambda x: x[1], reverse=True)[:3]
                    _drag_text = ", ".join([f"**{n}** ({w:.1f}%)" for n, w in _top_drags])
                    st.markdown(
                        f"**Priority:** {len(losers)} holding{'s' if len(losers) != 1 else ''} posted negative 5-year returns, "
                        f"representing **{_loser_wt * 100:.1f}% of eligible AUM**. "
                        f"Largest drag{'s' if len(_top_drags) > 1 else ''}: {_drag_text}."
                    )
                else:
                    st.markdown("**Priority:** All eligible holdings delivered positive 5-year returns.")

                if health >= 75:
                    st.success("Portfolio is in **strong health**. Focus on income growth and selective concentration reduction.")
                elif health >= 50:
                    st.warning("Portfolio health is **adequate**. Address capital drag from underperformers before they become material.")
                else:
                    st.error("Portfolio health requires **active attention**. Capital quality or income coverage is materially impaired.")
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
                                h = yf.download(_si(tick), start=(datetime.now() - timedelta(days=y * 365)), end=datetime.now(), auto_adjust=True, progress=False)
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
                                h = yf.download(_si(tick), start=(datetime.now() - timedelta(days=y * 365)), end=datetime.now(), auto_adjust=True, progress=False)
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
                sim_buy_curr = st.multiselect("Select current assets to acquire", options=df['Security'].unique(), default=winners[:2] if len(winners) > 1 else None)
                _new_asset_options = sorted([k for k in MASTER_INTEL if k not in df['Security'].values])
                sim_buy_new = st.multiselect("Select new assets to acquire", options=_new_asset_options)
            sim_buy = list(sim_buy_curr) + list(sim_buy_new)
            _alloc_pcts = {}
            if sim_buy:
                _n_buy = len(sim_buy)
                _default_pct = round(100.0 / _n_buy, 1)
                st.markdown("**Allocation of Freed Capital (%)**")
                _alloc_cols = st.columns(min(_n_buy, 4))
                for _i, _b in enumerate(sim_buy):
                    with _alloc_cols[_i % min(_n_buy, 4)]:
                        _alloc_pcts[_b] = st.number_input(f"{_b}", min_value=0.0, max_value=100.0, value=_default_pct, step=1.0, key=f"alloc_{_b}")
                _total_alloc_display = sum(_alloc_pcts.values())
                if abs(_total_alloc_display - 100.0) > 0.5:
                    st.warning(f"⚠️ Allocations sum to {_total_alloc_display:.1f}% — will be normalised proportionally on simulation.")
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
                    _sim_total_alloc = sum(_alloc_pcts.values()) or 1.0
                    _capital_alloc = {b: freed_capital * (_alloc_pcts.get(b, 0) / _sim_total_alloc) for b in sim_buy}
                    with st.spinner("Fetching live market yields for acquisition targets..."):
                        for b in sim_buy:
                            _b_capital = _capital_alloc[b]
                            if b in df['Security'].values:
                                tick = df[df['Security'] == b]['Ticker'].iloc[0]
                                dps = df[df['Security'] == b]['DPS'].iloc[0]
                            else:
                                _mi = MASTER_INTEL.get(b, {})
                                tick = _mi.get('Ticker')
                                dps = _mi.get('Rate', 0)
                            try:
                                if pd.notnull(tick):
                                    h = yf.download(_si(tick), period="5d", progress=False)
                                    c = h['Close'].iloc[:, 0] if isinstance(h['Close'], pd.DataFrame) else h['Close']
                                    new_income_added += _b_capital * (dps / float(c.iloc[-1]))
                                else:
                                    new_income_added += _b_capital * 0.04
                            except Exception:
                                new_income_added += _b_capital * 0.04
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

                    # ── QUANTITATIVE RISK METRICS BEFORE vs AFTER ─────────────────────
                    st.markdown("#### 📊 Quantitative Risk Metrics — Before vs After Rebalance")
                    _elig_sim = df[df['Ticker'].notnull()].copy()
                    if not _elig_sim.empty:
                        _tickers_b = ",".join(_elig_sim['Ticker'].tolist())
                        _names_b   = "|".join(_elig_sim['Security'].tolist())
                        _df_after  = _elig_sim[~_elig_sim['Security'].isin(sim_sell)].copy()
                        for _ba in sim_buy_curr:
                            _m = _df_after['Security'] == _ba
                            if _m.any():
                                _df_after.loc[_m, 'AUM (SGD)'] = _df_after.loc[_m, 'AUM (SGD)'] + _capital_alloc.get(_ba, 0)
                        for _ba in sim_buy_new:
                            _mi_new = MASTER_INTEL.get(_ba, {})
                            _new_tick = _mi_new.get('Ticker')
                            if _new_tick:
                                _new_row = pd.DataFrame([{'Security': _ba, 'Ticker': _new_tick, 'AUM (SGD)': _capital_alloc.get(_ba, 0)}])
                                _df_after = pd.concat([_df_after, _new_row], ignore_index=True)
                        if not _df_after.empty:
                            _tickers_a = ",".join(_df_after['Ticker'].tolist())
                            _names_a   = "|".join(_df_after['Security'].tolist())
                            with st.spinner("Computing risk metrics for simulation..."):
                                _rm_b = compute_risk_metrics(_tickers_b, _names_b)
                                _rm_a = compute_risk_metrics(_tickers_a, _names_a)
                            if not _rm_b.empty and not _rm_a.empty:
                                _rm_b = _rm_b.merge(_elig_sim[['Security', 'AUM (SGD)']], on='Security', how='left')
                                _rm_a = _rm_a.merge(_df_after[['Security', 'AUM (SGD)']], on='Security', how='left')

                                def _wavg(rdf, col):
                                    v = rdf.dropna(subset=[col, 'AUM (SGD)'])
                                    if v.empty: return None
                                    return float((v[col] * v['AUM (SGD)']).sum() / v['AUM (SGD)'].sum())

                                def _dvar_sum(rdf):
                                    v = rdf.dropna(subset=['Daily VaR 95% (%)', 'AUM (SGD)'])
                                    return float((v['Daily VaR 95% (%)'].abs() / 100 * v['AUM (SGD)']).sum()) if not v.empty else None

                                def _risk_row(icon, label, b_str, a_str, d_str, is_good, is_bad):
                                    if is_good:
                                        dc, arr = "#2ecc71", "▲"
                                    elif is_bad:
                                        dc, arr = "#e74c3c", "▼"
                                    else:
                                        dc, arr = "#8fa8cc", "◆"
                                    return (
                                        f"<tr style='border-bottom:1px solid #1a3354;'>"
                                        f"<td style='padding:9px 14px;font-weight:600;color:#dce8f5;white-space:nowrap;'>{icon}&nbsp;{label}</td>"
                                        f"<td style='padding:9px 14px;text-align:center;color:#8fa8cc;'>{b_str}</td>"
                                        f"<td style='padding:9px 14px;text-align:center;color:#f0d060;font-weight:700;'>{a_str}</td>"
                                        f"<td style='padding:9px 14px;text-align:center;color:{dc};font-weight:700;'>{arr}&nbsp;{d_str}</td>"
                                        f"</tr>"
                                    )

                                def _section_sep(title):
                                    return (
                                        f"<tr><td colspan='4' style='padding:5px 14px 3px;background:#0c1e33;"
                                        f"font-size:0.7rem;color:#4a6a8a;text-transform:uppercase;"
                                        f"letter-spacing:0.08em;font-weight:700;'>{title}</td></tr>"
                                    )

                                _rows_html = ""
                                # ── Income & Yield ─────────────────────
                                _rows_html += _section_sep("Income Impact")
                                _d_inc = new_inc - curr_inc
                                _d_yld = new_yield - curr_yield
                                _rows_html += _risk_row("💵", "Annual Income (SGD)",
                                    f"S${curr_inc:,.0f}", f"S${new_inc:,.0f}",
                                    f"S${abs(_d_inc):,.0f}", _d_inc > 0, _d_inc < 0)
                                _rows_html += _risk_row("📊", "Portfolio Yield",
                                    f"{curr_yield:.2f}%", f"{new_yield:.2f}%",
                                    f"{abs(_d_yld):.2f}%", _d_yld > 0, _d_yld < 0)

                                # ── Risk Metrics ───────────────────────
                                _rows_html += _section_sep("Risk Metrics — AUM-weighted averages")
                                _spec = [
                                    ("📈", "Ann. Return",      "Ann. Return (%)",     "%",  False, False),
                                    ("📉", "Ann. Volatility",  "Ann. Volatility (%)", "%",  False, True),
                                    ("⚡", "Sharpe Ratio",     "Sharpe Ratio",        "×",  False, False),
                                    ("📐", "Beta (vs STI)",    "Beta (vs STI)",       "×",  False, None),
                                    ("⚠️", "Daily VaR 95%",   "Daily VaR 95% (%)",  "%",  True,  True),
                                    ("🕳️", "Max Drawdown",    "Max Drawdown (%)",    "%",  True,  True),
                                ]
                                for _ico, _lbl, _col, _unit, _abs, _lower in _spec:
                                    _bv = _wavg(_rm_b, _col)
                                    _av = _wavg(_rm_a, _col)
                                    if _bv is None or _av is None:
                                        continue
                                    _bvd = abs(_bv) if _abs else _bv
                                    _avd = abs(_av) if _abs else _av
                                    _dv  = _avd - _bvd
                                    if _lower is True:
                                        _g, _bd = _dv < 0, _dv > 0
                                    elif _lower is False:
                                        _g, _bd = _dv > 0, _dv < 0
                                    else:
                                        _g, _bd = False, False
                                    if _unit == "×":
                                        _bs = f"{_bvd:.3f}"
                                        _as = f"{_avd:.3f}"
                                        _ds = f"{abs(_dv):.3f}"
                                    else:
                                        _bs = f"{_bvd:.2f}%"
                                        _as = f"{_avd:.2f}%"
                                        _ds = f"{abs(_dv):.2f}%"
                                    _rows_html += _risk_row(_ico, _lbl, _bs, _as, _ds, _g, _bd)

                                # Dollar VaR
                                _bv_dv = _dvar_sum(_rm_b)
                                _av_dv = _dvar_sum(_rm_a)
                                if _bv_dv is not None and _av_dv is not None:
                                    _dd = _av_dv - _bv_dv
                                    _rows_html += _risk_row("💰", "Dollar VaR 95%",
                                        f"S${_bv_dv:,.0f}", f"S${_av_dv:,.0f}",
                                        f"S${abs(_dd):,.0f}", _dd < 0, _dd > 0)

                                st.markdown(f"""
                                <div style='margin-top:18px;border-radius:10px;overflow:hidden;
                                            border:1px solid #1a3354;
                                            box-shadow:0 3px 14px rgba(0,0,0,0.35);'>
                                  <table style='width:100%;border-collapse:collapse;font-size:0.87rem;'>
                                    <thead>
                                      <tr style='background:linear-gradient(90deg,#0c1e33,#0f2540);'>
                                        <th style='padding:11px 14px;text-align:left;color:#6a93b8;
                                                   font-size:0.73rem;text-transform:uppercase;
                                                   letter-spacing:0.08em;border-bottom:2px solid #1a3354;
                                                   width:38%;'>Metric</th>
                                        <th style='padding:11px 14px;text-align:center;color:#6a93b8;
                                                   font-size:0.73rem;text-transform:uppercase;
                                                   letter-spacing:0.08em;border-bottom:2px solid #1a3354;'>Before</th>
                                        <th style='padding:11px 14px;text-align:center;color:#c8a400;
                                                   font-size:0.73rem;text-transform:uppercase;
                                                   letter-spacing:0.08em;border-bottom:2px solid #1a3354;'>After</th>
                                        <th style='padding:11px 14px;text-align:center;color:#6a93b8;
                                                   font-size:0.73rem;text-transform:uppercase;
                                                   letter-spacing:0.08em;border-bottom:2px solid #1a3354;'>Change</th>
                                      </tr>
                                    </thead>
                                    <tbody style='background:#091828;'>
                                      {_rows_html}
                                    </tbody>
                                  </table>
                                </div>
                                <p style='margin-top:7px;font-size:0.71rem;color:#4a6a8a;'>
                                  ▲&nbsp;Green&nbsp;=&nbsp;improved &nbsp;|&nbsp;
                                  ▼&nbsp;Red&nbsp;=&nbsp;worsened &nbsp;|&nbsp;
                                  Risk metrics are AUM-weighted averages across ticker-eligible holdings only.
                                  VaR = est. max 1-day loss at 95% confidence. Max Drawdown and VaR shown as absolute values.
                                </p>
                                """, unsafe_allow_html=True)

            # ── SENTIMENT RADAR ──────────────────────
            st.markdown("---")
            st.markdown("### 🧠 Sentiment Radar — FinBERT News Analysis")
            st.markdown(
                '<p class="tab-subtitle" style="margin-top:-10px;">AI-scored sentiment from the latest news headlines for each holding, '
                'powered by <strong>ProsusAI/FinBERT</strong> via HuggingFace.</p>',
                unsafe_allow_html=True,
            )
            hf_token = st.session_state.get("hf_token", "")
            if not hf_token:
                st.info("Enter your HuggingFace API token in the **AI Sentiment (FinBERT)** panel in the sidebar to activate this section.")
            else:
                sent_eligible = df[df['Ticker'].notnull()].copy()
                if sent_eligible.empty:
                    st.warning("No holdings with market tickers available for sentiment analysis.")
                else:
                    col_run, col_clear = st.columns([1, 5])
                    with col_run:
                        run_sentiment = st.button("Analyse Sentiment", type="primary")
                    with col_clear:
                        if st.button("Clear Results"):
                            st.session_state.pop("sentiment_results", None)
                            st.rerun()

                    if run_sentiment:
                        all_results = {}
                        loading_flag = False
                        progress_bar = st.progress(0, text="Fetching news and scoring sentiment...")
                        holdings = sent_eligible[['Security', 'Ticker', 'AUM (SGD)']].values.tolist()
                        for idx, (name, ticker, aum) in enumerate(holdings):
                            headlines = _fetch_news_headlines(_si(ticker))
                            if not headlines:
                                all_results[name] = {
                                    "ticker": ticker, "aum": aum,
                                    "headlines": [], "scores": [],
                                    "summary": {"label": "neutral", "net": 0.0,
                                                "pos": 0, "neu": 0, "neg": 0, "count": 0},
                                }
                                progress_bar.progress((idx + 1) / len(holdings),
                                                      text=f"No news found for {name} — skipping")
                                continue
                            scored = _call_finbert(headlines, hf_token)
                            if scored == "loading":
                                loading_flag = True
                                break
                            if scored is None:
                                scored = [{"label": "neutral", "score": 1.0}] * len(headlines)
                            all_results[name] = {
                                "ticker": ticker, "aum": aum,
                                "headlines": headlines, "scores": scored,
                                "summary": _aggregate_sentiment(scored),
                            }
                            progress_bar.progress((idx + 1) / len(holdings),
                                                  text=f"Scored {name} — {len(headlines)} headline(s)")
                        progress_bar.empty()
                        if loading_flag:
                            st.warning("FinBERT model is loading on HuggingFace — wait ~20 seconds and click Analyse Sentiment again.")
                        else:
                            st.session_state["sentiment_results"] = all_results

                    if "sentiment_results" in st.session_state:
                        results = st.session_state["sentiment_results"]

                        # ── KPI summary bar ──────────────────────────────
                        n_bull = sum(1 for r in results.values() if r["summary"]["label"] == "positive")
                        n_bear = sum(1 for r in results.values() if r["summary"]["label"] == "negative")
                        n_neut = sum(1 for r in results.values() if r["summary"]["label"] == "neutral")
                        total_scored = len(results)
                        aum_bull = sum(r["aum"] for r in results.values() if r["summary"]["label"] == "positive")
                        aum_bear = sum(r["aum"] for r in results.values() if r["summary"]["label"] == "negative")
                        total_aum_sent = sum(r["aum"] for r in results.values())
                        wtd_net = (
                            sum(r["summary"]["net"] * r["aum"] for r in results.values()) / total_aum_sent
                            if total_aum_sent > 0 else 0.0
                        )
                        port_label = "positive" if wtd_net > 0.1 else ("negative" if wtd_net < -0.1 else "neutral")
                        sk1, sk2, sk3, sk4 = st.columns(4)
                        with sk1:
                            kpi_card(_SENTIMENT_ICON[port_label],
                                     "Portfolio Sentiment",
                                     _SENTIMENT_LABEL[port_label])
                        with sk2:
                            kpi_card("📈", "Bullish Holdings",
                                     f"{n_bull} ({aum_bull / total_aum_sent * 100:.0f}% AUM)" if total_aum_sent else str(n_bull))
                        with sk3:
                            kpi_card("📉", "Bearish Holdings",
                                     f"{n_bear} ({aum_bear / total_aum_sent * 100:.0f}% AUM)" if total_aum_sent else str(n_bear))
                        with sk4:
                            kpi_card("➡️", "Neutral Holdings", str(n_neut))

                        st.markdown("<br>", unsafe_allow_html=True)

                        # ── Per-holding results table ─────────────────────
                        rows_html = ""
                        for name, r in sorted(results.items(),
                                              key=lambda x: x[1]["summary"]["net"],
                                              reverse=True):
                            s   = r["summary"]
                            col = _SENTIMENT_COLOUR[s["label"]]
                            lbl = _SENTIMENT_LABEL[s["label"]]
                            ico = _SENTIMENT_ICON[s["label"]]
                            bar_pos = int(s["pos"] / s["count"] * 100) if s["count"] else 0
                            bar_neg = int(s["neg"] / s["count"] * 100) if s["count"] else 0
                            bar_neu = 100 - bar_pos - bar_neg
                            no_news = s["count"] == 0
                            rows_html += f"""
                            <tr>
                              <td style='padding:8px 12px;font-weight:600;'>{name}</td>
                              <td style='padding:8px 12px;color:#8fa8cc;font-size:0.82rem;'>{r['ticker']}</td>
                              <td style='padding:8px 12px;'>
                                <span style='background:{col};color:#fff;padding:3px 10px;
                                             border-radius:12px;font-size:0.8rem;font-weight:600;'>
                                  {ico} {lbl}
                                </span>
                              </td>
                              <td style='padding:8px 12px;min-width:140px;'>
                                {'<span style="color:#8fa8cc;font-size:0.8rem;">No news found</span>' if no_news else
                                 f'<div style="display:flex;height:10px;border-radius:5px;overflow:hidden;width:140px;">'
                                 f'<div style="width:{bar_pos}%;background:#27AE60;"></div>'
                                 f'<div style="width:{bar_neu}%;background:#7F8C8D;"></div>'
                                 f'<div style="width:{bar_neg}%;background:#E74C3C;"></div>'
                                 f'</div>'
                                 f'<div style="font-size:0.72rem;color:#8fa8cc;margin-top:3px;">'
                                 f'{s["pos"]}↑ {s["neu"]}→ {s["neg"]}↓ of {s["count"]} headlines</div>'}
                              </td>
                              <td style='padding:8px 12px;font-size:0.8rem;color:#8fa8cc;'>
                                {f"{s['net']:+.2f}" if not no_news else "—"}
                              </td>
                            </tr>"""
                        st.markdown(f"""
                        <table style='width:100%;border-collapse:collapse;font-size:0.88rem;'>
                          <thead>
                            <tr style='border-bottom:2px solid #1e3a5f;color:#8fa8cc;font-size:0.78rem;text-transform:uppercase;'>
                              <th style='padding:6px 12px;text-align:left;'>Holding</th>
                              <th style='padding:6px 12px;text-align:left;'>Ticker</th>
                              <th style='padding:6px 12px;text-align:left;'>Signal</th>
                              <th style='padding:6px 12px;text-align:left;'>Headline Breakdown</th>
                              <th style='padding:6px 12px;text-align:left;'>Net Score</th>
                            </tr>
                          </thead>
                          <tbody>{rows_html}</tbody>
                        </table>
                        """, unsafe_allow_html=True)

                        # ── Headline drill-down ───────────────────────────
                        st.markdown("<br>", unsafe_allow_html=True)
                        with st.expander("Headline Details — click to inspect per-holding news"):
                            drill = st.selectbox(
                                "Select holding",
                                options=[n for n, r in results.items() if r["headlines"]],
                                key="sent_drill",
                            )
                            if drill and drill in results:
                                dr = results[drill]
                                for headline, score in zip(dr["headlines"], dr["scores"]):
                                    c   = _SENTIMENT_COLOUR[score["label"]]
                                    lbl = _SENTIMENT_LABEL[score["label"]]
                                    st.markdown(
                                        f"<div style='padding:7px 12px;margin:4px 0;border-left:4px solid {c};"
                                        f"background:rgba(255,255,255,0.03);border-radius:0 6px 6px 0;'>"
                                        f"<span style='color:{c};font-weight:600;font-size:0.8rem;'>{lbl} {score['score']:.0%}</span>"
                                        f"<div style='margin-top:3px;font-size:0.88rem;'>{headline}</div>"
                                        f"</div>",
                                        unsafe_allow_html=True,
                                    )
                        st.caption(
                            "Sentiment scored by **ProsusAI/FinBERT** via HuggingFace Inference API. "
                            "Net score = (bullish − bearish) / headline count. "
                            "Results cached for 1 hour — click **Analyse Sentiment** to refresh."
                        )

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

                    # --- Portfolio Risk Commentary (technical / plain-English toggle) ---
                    _avg_vol    = risk_df['Ann. Volatility (%)'].mean()
                    _avg_sharpe = risk_df['Sharpe Ratio'].mean()
                    _avg_beta   = beta_vals.mean() if not beta_vals.empty else None
                    _total_aum  = risk_m['AUM (SGD)'].sum()
                    _var_pct    = (total_var / _total_aum * 100) if _total_aum > 0 else 0
                    _valid_s    = risk_df.dropna(subset=['Sharpe Ratio'])
                    _valid_dd   = risk_df.dropna(subset=['Max Drawdown (%)'])
                    _best_s     = _valid_s.loc[_valid_s['Sharpe Ratio'].idxmax()] if not _valid_s.empty else None
                    _worst_dd   = _valid_dd.loc[_valid_dd['Max Drawdown (%)'].idxmin()] if not _valid_dd.empty else None

                    if 'risk_commentary_simple' not in st.session_state:
                        st.session_state['risk_commentary_simple'] = False

                    _simple = st.toggle(
                        "Plain-language explanation",
                        key="risk_commentary_simple",
                        help="Switch between professional commentary and a plain-English summary"
                    )

                    if not _simple:
                        # ── Technical version ──────────────────────────────────
                        if _avg_vol < 10:
                            _vol_lbl = "low"; _vol_col = "#27ae60"
                            _vol_ctx = "consistent with a capital-preservation or income-focused mandate"
                        elif _avg_vol < 20:
                            _vol_lbl = "moderate"; _vol_col = "#e67e22"
                            _vol_ctx = "typical of a diversified SGX equity portfolio"
                        else:
                            _vol_lbl = "elevated"; _vol_col = "#e74c3c"
                            _vol_ctx = "indicating meaningful price-swing risk that merits close monitoring"

                        if _avg_sharpe > 1.5:
                            _sharpe_ctx = f"an <strong>excellent</strong> risk-adjusted return profile (Sharpe {_avg_sharpe:.2f}) — well above the 1.0 practitioner benchmark, implying each unit of risk is being well rewarded"
                        elif _avg_sharpe > 1.0:
                            _sharpe_ctx = f"a <strong>good</strong> risk-adjusted return profile (Sharpe {_avg_sharpe:.2f}), clearing the 1.0 threshold considered the minimum acceptable by most institutions"
                        elif _avg_sharpe > 0.5:
                            _sharpe_ctx = f"a <strong>fair</strong> risk-adjusted return profile (Sharpe {_avg_sharpe:.2f}) — returns are positive but the compensation for risk taken is below optimal; selective reallocation may improve efficiency"
                        else:
                            _sharpe_ctx = f"a <strong>weak</strong> risk-adjusted return profile (Sharpe {_avg_sharpe:.2f}), suggesting the portfolio is generating insufficient return relative to the volatility carried"

                        if _avg_beta is None:
                            _beta_ctx = "market-sensitivity data is unavailable for this portfolio"
                        elif _avg_beta < 0.8:
                            _beta_ctx = f"a <strong>defensive</strong> market posture (β {_avg_beta:.2f}) — the portfolio is expected to decline less than the STI in broad market sell-offs, favouring capital preservation over cyclical upside"
                        elif _avg_beta <= 1.2:
                            _beta_ctx = f"<strong>market-neutral</strong> sensitivity to the STI (β {_avg_beta:.2f}), broadly tracking the local index with no pronounced defensive or aggressive tilt"
                        else:
                            _beta_ctx = f"an <strong>aggressive</strong> market bias (β {_avg_beta:.2f}) — amplifying both gains and drawdowns relative to the STI; suitable only for investors with a high risk tolerance and longer time horizon"

                        _tail_para = ""
                        if _best_s is not None and _worst_dd is not None:
                            _tail_para = (
                                f"<p style='font-size:0.85rem;color:#2c3e50;margin:0;line-height:1.65;'>"
                                f"At the individual holding level, <strong>{_best_s['Security']}</strong> delivers the strongest risk-adjusted return "
                                f"(Sharpe {_best_s['Sharpe Ratio']:.2f}), making it the portfolio's most efficient income-to-risk contributor. "
                                f"Conversely, <strong>{_worst_dd['Security']}</strong> exhibits the deepest historical peak-to-trough drawdown "
                                f"({_worst_dd['Max Drawdown (%)']:.1f}%), warranting a position-size review against the investor's maximum tolerable loss threshold."
                                f"</p>"
                            )

                        _html = f"""
<div style='background:rgba(26,82,118,0.05);border-left:4px solid #1a5276;
            border-radius:0 8px 8px 0;padding:16px 20px;margin:6px 0 4px 0;'>
  <p style='font-size:0.82rem;font-weight:700;color:#1a5276;margin:0 0 10px 0;
            letter-spacing:0.05em;text-transform:uppercase;'>Portfolio Risk Commentary</p>
  <p style='font-size:0.85rem;color:#2c3e50;margin:0 0 9px 0;line-height:1.65;'>
    The portfolio exhibits <strong style='color:{_vol_col};'>{_vol_lbl} average annualised volatility of {_avg_vol:.1f}%</strong>,
    {_vol_ctx}. On a risk-adjusted basis, the holdings present {_sharpe_ctx}.
  </p>
  <p style='font-size:0.85rem;color:#2c3e50;margin:0 0 9px 0;line-height:1.65;'>
    In terms of market sensitivity, the portfolio carries {_beta_ctx}.
    At a 95% confidence level, the estimated single-day Value at Risk stands at
    <strong>S${total_var:,.0f}</strong> ({_var_pct:.2f}% of invested capital),
    implying that on approximately 1 in 20 trading sessions, mark-to-market losses could approach
    or exceed this level under normal market conditions.
  </p>
  {_tail_para}
</div>"""

                    else:
                        # ── Plain-English version ──────────────────────────────
                        if _avg_vol < 10:
                            _s_vol = f"your portfolio is quite <strong style='color:#27ae60;'>steady</strong> — prices don't jump around much (average swing: {_avg_vol:.1f}% per year)"
                        elif _avg_vol < 20:
                            _s_vol = f"your portfolio has a <strong style='color:#e67e22;'>moderate amount of price movement</strong> — values can shift noticeably from month to month (average swing: {_avg_vol:.1f}% per year)"
                        else:
                            _s_vol = f"your portfolio is <strong style='color:#e74c3c;'>quite bumpy</strong> — expect significant ups and downs in value (average swing: {_avg_vol:.1f}% per year)"

                        if _avg_sharpe > 1.5:
                            _s_sharpe = f"you are being <strong>very well rewarded</strong> for the risk you're taking — the returns are strong relative to the volatility (score: {_avg_sharpe:.2f})"
                        elif _avg_sharpe > 1.0:
                            _s_sharpe = f"the returns are <strong>worth the risk</strong> — for every unit of price swing, you are earning a solid return (score: {_avg_sharpe:.2f})"
                        elif _avg_sharpe > 0.5:
                            _s_sharpe = f"the returns are <strong>okay but not great</strong> for the risk involved — there may be room to improve by switching some holdings (score: {_avg_sharpe:.2f})"
                        else:
                            _s_sharpe = f"the returns <strong>do not justify the price swings</strong> at the moment — it may be worth reviewing whether the higher-risk holdings are pulling their weight (score: {_avg_sharpe:.2f})"

                        if _avg_beta is None:
                            _s_beta = "we don't have enough data to compare your portfolio against the Singapore market"
                        elif _avg_beta < 0.8:
                            _s_beta = f"your portfolio tends to <strong>fall less than the Singapore market</strong> when things turn bad — it is on the defensive side (market sensitivity: {_avg_beta:.2f})"
                        elif _avg_beta <= 1.2:
                            _s_beta = f"your portfolio moves <strong>roughly in line with the Singapore market</strong> — neither particularly defensive nor aggressive (market sensitivity: {_avg_beta:.2f})"
                        else:
                            _s_beta = f"your portfolio tends to <strong>swing more than the Singapore market</strong> — bigger gains when markets rise, but bigger drops when they fall (market sensitivity: {_avg_beta:.2f})"

                        _s_tail = ""
                        if _best_s is not None and _worst_dd is not None:
                            _s_tail = (
                                f"<p style='font-size:0.85rem;color:#2c3e50;margin:0;line-height:1.65;'>"
                                f"Among your holdings, <strong>{_best_s['Security']}</strong> gives you the best bang for your buck — "
                                f"it delivers solid returns without taking on too much risk. "
                                f"On the other hand, <strong>{_worst_dd['Security']}</strong> has had the biggest price drop from its peak "
                                f"({abs(_worst_dd['Max Drawdown (%)']):.1f}% at its worst), so it's worth keeping an eye on how much of your money is sitting there."
                                f"</p>"
                            )

                        _html = f"""
<div style='background:rgba(39,174,96,0.05);border-left:4px solid #27ae60;
            border-radius:0 8px 8px 0;padding:16px 20px;margin:6px 0 4px 0;'>
  <p style='font-size:0.82rem;font-weight:700;color:#1e8449;margin:0 0 10px 0;
            letter-spacing:0.05em;text-transform:uppercase;'>Plain-Language Summary</p>
  <p style='font-size:0.85rem;color:#2c3e50;margin:0 0 9px 0;line-height:1.65;'>
    In terms of how much your investments move up and down in price, {_s_vol}.
    When it comes to whether the returns are worth it, {_s_sharpe}.
  </p>
  <p style='font-size:0.85rem;color:#2c3e50;margin:0 0 9px 0;line-height:1.65;'>
    As for how closely your portfolio follows the Singapore stock market, {_s_beta}.
    On a typical bad day — the kind that happens roughly once a month —
    your portfolio could drop by around <strong>S${total_var:,.0f}</strong>
    ({_var_pct:.2f}% of your total invested amount). This is just an estimate, not a guarantee.
  </p>
  {_s_tail}
</div>"""

                    st.markdown(_html, unsafe_allow_html=True)
                    # --- end commentary ---

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
                        fig_cal.update_layout(barmode='stack', template='plotly_white', height=460,
                                              xaxis_title="Month", yaxis_title="Expected Income (SGD)",
                                              legend=dict(orientation="h", yanchor="bottom", y=1.08,
                                                          xanchor="left", x=0, font=dict(size=11)),
                                              margin=dict(t=100, b=40, l=20, r=20))
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
            # DIVIDEND CONSISTENCY TRACKER
            # ══════════════════════════════════════════
            st.markdown("---")
            st.markdown("### 🔍 Dividend Consistency Tracker")
            st.markdown(
                '<p style="font-size:0.88rem;color:#555;margin-top:-6px;">'
                'Analyse the historical reliability, growth trajectory, and free cash flow coverage '
                'of any holding\'s dividend record.</p>',
                unsafe_allow_html=True
            )
            _dct_elig = df[df['Ticker'].notnull()].copy()
            if _dct_elig.empty:
                st.info("No holdings with market tickers available for consistency analysis.")
            else:
                _dct_opts = sorted(_dct_elig['Security'].tolist())
                _dct_c1, _dct_c2 = st.columns([3, 1])
                with _dct_c1:
                    _dct_stock = st.selectbox("Select a holding to analyse", options=_dct_opts, key="dct_sel")
                with _dct_c2:
                    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                    _dct_go = st.button("🔍 Analyse", key="dct_go")

                if _dct_go:
                    _dct_row  = _dct_elig[_dct_elig['Security'] == _dct_stock].iloc[0]
                    _dct_tick = _dct_row['Ticker']
                    _dct_dps  = float(_dct_row['DPS'])

                    with st.spinner(f"Fetching dividend and financial data for {_dct_stock}…"):
                        try:
                            _tk       = yf.Ticker(_si(_dct_tick))
                            _div_raw  = _tk.dividends
                            _cf_stmt  = _tk.cashflow
                            _tk_info  = _tk.info
                        except Exception as _fe:
                            st.error(f"Data fetch failed: {_fe}")
                            _div_raw  = pd.Series(dtype=float)
                            _cf_stmt  = pd.DataFrame()
                            _tk_info  = {}

                    if _div_raw.empty:
                        st.warning(f"No dividend history found on Yahoo Finance for **{_dct_stock}**.")
                    else:
                        # ── Normalise timezone ────────────────────────────────
                        if hasattr(_div_raw.index, 'tz') and _div_raw.index.tz:
                            _div_raw.index = _div_raw.index.tz_localize(None)

                        _annual_div = _div_raw.groupby(_div_raw.index.year).sum()
                        _yr_list    = _annual_div.index.tolist()
                        _n_yrs      = len(_yr_list)
                        _curr_yr    = datetime.now().year

                        # ── Consecutive-year longevity ────────────────────────
                        _longevity = 1
                        if _n_yrs >= 2:
                            for _li in range(len(_yr_list) - 1, 0, -1):
                                if _yr_list[_li] - _yr_list[_li - 1] <= 1:
                                    _longevity += 1
                                else:
                                    break

                        # ── YoY growth ────────────────────────────────────────
                        _yoy      = (_annual_div.pct_change() * 100).round(2)
                        _yoy_val  = _yoy.dropna()
                        _neg_yrs  = int((_yoy_val < 0).sum())
                        _pos_yrs  = int((_yoy_val > 0).sum())

                        # ── 5Y CAGR ───────────────────────────────────────────
                        _cagr_5y  = None
                        _5y_data  = _annual_div[_annual_div.index >= _curr_yr - 5]
                        if len(_5y_data) >= 2 and float(_5y_data.iloc[0]) > 0:
                            _5y_span = _5y_data.index[-1] - _5y_data.index[0]
                            if _5y_span > 0:
                                _cagr_5y = ((_5y_data.iloc[-1] / _5y_data.iloc[0]) ** (1 / _5y_span) - 1) * 100

                        # Full-history CAGR as fallback
                        _cagr_all = None
                        if _n_yrs >= 3 and float(_annual_div.iloc[0]) > 0:
                            _full_span = _yr_list[-1] - _yr_list[0]
                            if _full_span > 0:
                                _cagr_all = ((_annual_div.iloc[-1] / _annual_div.iloc[0]) ** (1 / _full_span) - 1) * 100
                        _cagr_use = _cagr_5y if _cagr_5y is not None else _cagr_all

                        # ── FCF Payout Ratio ──────────────────────────────────
                        _fcf_payout = None
                        _fcf_note   = ""
                        try:
                            _shares   = _tk_info.get('sharesOutstanding') or 0
                            _fcf_abs  = _tk_info.get('freeCashflow') or 0
                            _lat_dps  = float(_annual_div.iloc[-1]) if _n_yrs > 0 else _dct_dps
                            _div_est  = _lat_dps * _shares

                            if _fcf_abs > 0 and _shares > 0:
                                _fcf_payout = (_div_est / _fcf_abs) * 100
                                _fcf_note   = "trailing 12M FCF (Yahoo Finance)"
                            elif not _cf_stmt.empty:
                                _cf_idx_l = [str(x).lower() for x in _cf_stmt.index]

                                def _cf_row(keys):
                                    for _k in keys:
                                        _hits = [_i for _i, _x in enumerate(_cf_idx_l) if _k in _x]
                                        if _hits:
                                            _v = _cf_stmt.iloc[_hits[0], 0]
                                            return float(_v) if pd.notna(_v) else None
                                    return None

                                _fcf_val = _cf_row(['free cash flow'])
                                if _fcf_val is None:
                                    _ocf = _cf_row(['operating cash flow', 'cash from operating'])
                                    _cap = _cf_row(['capital expenditure'])
                                    if _ocf is not None and _cap is not None:
                                        _fcf_val = _ocf + _cap
                                _div_paid = _cf_row(['dividends paid', 'cash dividends', 'common stock dividend'])

                                if _fcf_val and _fcf_val > 0:
                                    _num = abs(_div_paid) if _div_paid else _div_est
                                    _fcf_payout = (_num / _fcf_val) * 100
                                    _fcf_note   = "cash flow statement" if _div_paid else "estimated from DPS × shares"
                        except Exception:
                            pass

                        # ── Consistency score (100 pts, 4 components of 25 each) ──
                        _sc_lon = min(25.0, (_longevity / 10) * 25)
                        _sc_grw = 25.0 * min(1.0, max(0.0, (_cagr_use or 0) / 8)) if _cagr_use is not None else 0.0
                        _n_per  = max(1, _n_yrs - 1)
                        _sc_rel = 25.0 * max(0.0, 1.0 - _neg_yrs / _n_per)
                        if _fcf_payout is not None:
                            _sc_fcf = (25.0 if _fcf_payout <= 50
                                       else 25.0 * max(0.0, 1 - (_fcf_payout - 50) / 50) if _fcf_payout <= 100
                                       else 0.0)
                        else:
                            _sc_fcf = 12.5
                        _con_score = round(_sc_lon + _sc_grw + _sc_rel + _sc_fcf)

                        if   _con_score >= 80: _rating, _rcol = "AA — Highly Reliable",  "#1E8449"
                        elif _con_score >= 65: _rating, _rcol = "A  — Reliable",          "#27AE60"
                        elif _con_score >= 50: _rating, _rcol = "B  — Adequate",           "#B7770D"
                        elif _con_score >= 35: _rating, _rcol = "C  — Inconsistent",       "#E67E22"
                        else:                  _rating, _rcol = "D  — Unreliable",         "#C0392B"

                        # ── KPI row ───────────────────────────────────────────
                        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                        _k1, _k2, _k3, _k4 = st.columns(4)
                        with _k1:
                            kpi_card("📅", "Dividend Longevity", f"{_longevity} yrs")
                        with _k2:
                            kpi_card("📈", "5Y Dividend CAGR", f"{_cagr_5y:.1f}%" if _cagr_5y is not None else "N/A")
                        with _k3:
                            kpi_card("📉", "Years of Cuts", str(_neg_yrs))
                        with _k4:
                            kpi_card("💵", "FCF Payout Ratio", f"{min(_fcf_payout, 999):.0f}%" if _fcf_payout is not None else "N/A")

                        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

                        # ── Score gauge + component breakdown ─────────────────
                        _gs1, _gs2 = st.columns([1, 2])
                        with _gs1:
                            _gc  = "#1E8449" if _con_score >= 65 else "#B7770D" if _con_score >= 40 else "#C0392B"
                            _gfig = go.Figure(go.Indicator(
                                mode="gauge+number", value=_con_score,
                                number={"suffix": "/100", "font": {"size": 26, "color": _gc}},
                                title={"text": "Consistency Score", "font": {"size": 13, "color": "#555"}},
                                gauge={
                                    "axis": {"range": [0, 100], "tickcolor": "#aaa"},
                                    "bar":  {"color": _gc, "thickness": 0.28},
                                    "bgcolor": "white", "borderwidth": 0,
                                    "steps": [{"range": [0,  35], "color": "#fde8e8"},
                                              {"range": [35, 65], "color": "#fef9e7"},
                                              {"range": [65,100], "color": "#eafaf1"}]
                                }
                            ))
                            _gfig.update_layout(height=230, margin=dict(t=30, b=5, l=20, r=20), paper_bgcolor="white")
                            st.plotly_chart(_gfig, use_container_width=True)
                            st.markdown(
                                f"<div style='text-align:center;font-size:0.9rem;font-weight:700;"
                                f"color:{_rcol};margin-top:-14px;'>{_rating}</div>",
                                unsafe_allow_html=True
                            )

                        with _gs2:
                            def _dct_bar(s, mx, c):
                                p = int(s / mx * 100)
                                return (
                                    f"<div style='background:#e8edf2;border-radius:4px;height:7px;width:100%;'>"
                                    f"<div style='background:{c};width:{p}%;height:7px;border-radius:4px;'></div></div>"
                                )

                            _sc_rows = [
                                ("Longevity",     _sc_lon, 25,
                                 "#1E8449" if _sc_lon >= 17.5 else "#B7770D" if _sc_lon >= 10 else "#C0392B"),
                                ("Growth (CAGR)", _sc_grw, 25,
                                 "#1E8449" if _sc_grw >= 17.5 else "#B7770D" if _sc_grw >= 10 else "#C0392B"),
                                ("Reliability",   _sc_rel, 25,
                                 "#1E8449" if _sc_rel >= 17.5 else "#B7770D" if _sc_rel >= 10 else "#C0392B"),
                                ("FCF Coverage",  _sc_fcf, 25,
                                 "#1E8449" if _sc_fcf >= 17.5 else "#B7770D" if _sc_fcf >= 10 else "#C0392B"),
                            ]
                            _sc_tbl = (
                                "<table style='width:100%;border-collapse:collapse;font-size:0.82rem;margin-top:10px;'>"
                                "<tr>"
                                "<th style='text-align:left;color:#4a6a8a;padding:3px 8px;font-weight:600;'>Component</th>"
                                "<th style='text-align:right;color:#4a6a8a;padding:3px 8px;'>Score</th>"
                                "<th style='text-align:right;color:#4a6a8a;padding:3px 8px;'>Max</th>"
                                "<th style='padding:3px 8px;width:38%;'></th>"
                                "</tr>"
                            )
                            for _rl, _rs, _rm, _rc in _sc_rows:
                                _sc_tbl += (
                                    f"<tr>"
                                    f"<td style='padding:7px 8px;color:#1a2a3a;font-weight:600;'>{_rl}</td>"
                                    f"<td style='text-align:right;padding:7px 8px;color:{_rc};font-weight:700;'>{_rs:.1f}</td>"
                                    f"<td style='text-align:right;padding:7px 8px;color:#4a6a8a;'>{_rm}</td>"
                                    f"<td style='padding:7px 10px;vertical-align:middle;'>{_dct_bar(_rs, _rm, _rc)}</td>"
                                    f"</tr>"
                                )
                            _sc_tbl += "</table>"
                            st.markdown(_sc_tbl, unsafe_allow_html=True)

                            # ── Narrative ─────────────────────────────────────
                            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                            if _longevity >= 10:
                                st.markdown(f"• **{_longevity}-year** uninterrupted payment record signals strong institutional commitment to the dividend.")
                            elif _longevity >= 5:
                                st.markdown(f"• **{_longevity}-year** payment history is moderate; watch consistency across full business cycles.")
                            else:
                                st.markdown(f"• Only **{_longevity} year{'s' if _longevity != 1 else ''}** of history — too short to establish a reliable track record.")

                            if _cagr_5y is not None:
                                if _cagr_5y >= 5:
                                    st.markdown(f"• 5Y CAGR of **{_cagr_5y:.1f}%** — dividend growing well above inflation.")
                                elif _cagr_5y >= 0:
                                    st.markdown(f"• 5Y CAGR of **{_cagr_5y:.1f}%** — positive but modest; review whether growth keeps pace with inflation.")
                                else:
                                    st.markdown(f"• 5Y CAGR of **{_cagr_5y:.1f}%** — dividend eroding; verify against latest management guidance.")
                            elif _cagr_all is not None:
                                st.markdown(f"• Full-history CAGR of **{_cagr_all:.1f}%** (fewer than 5 complete years of data).")

                            if _neg_yrs > 0:
                                st.markdown(f"• **{_neg_yrs} cut{'s' if _neg_yrs != 1 else ''}** recorded over {_n_yrs} years; review timing against macro downturns or restructuring events.")
                            else:
                                st.markdown("• No dividend cuts recorded over the available history.")

                            if _fcf_payout is not None:
                                if _fcf_payout <= 60:
                                    st.markdown(f"• FCF payout of **{_fcf_payout:.0f}%** — dividends are comfortably covered by free cash flow.")
                                elif _fcf_payout <= 100:
                                    st.markdown(f"• FCF payout of **{_fcf_payout:.0f}%** is elevated; monitor cash generation for sustainability risks.")
                                else:
                                    st.markdown(f"• FCF payout of **{_fcf_payout:.0f}%** exceeds free cash flow — the dividend may not be self-funding from operations.")
                            else:
                                st.markdown("• FCF payout ratio unavailable — verify dividend sustainability from the latest annual report.")

                        # ── Side-by-side charts ───────────────────────────────
                        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
                        _cc1, _cc2 = st.columns(2)
                        with _cc1:
                            _ann_mean = float(_annual_div.mean())
                            _af = go.Figure(go.Bar(
                                x=[str(y) for y in _annual_div.index],
                                y=_annual_div.values,
                                marker_color=["#27AE60" if v >= _ann_mean else "#3498DB" for v in _annual_div.values],
                                text=[f"S${v:.3f}" for v in _annual_div.values],
                                textposition="outside",
                            ))
                            _af.update_layout(
                                title=dict(text=f"Annual Dividends Per Share — {_dct_stock}", font=dict(size=12)),
                                template="plotly_white", height=320,
                                xaxis=dict(title="Year", tickangle=-45),
                                yaxis=dict(title="DPS (S$)"),
                                margin=dict(t=50, b=50, l=10, r=10),
                                showlegend=False
                            )
                            st.plotly_chart(_af, use_container_width=True)

                        with _cc2:
                            if not _yoy_val.empty:
                                _yf2 = go.Figure(go.Bar(
                                    x=[str(y) for y in _yoy_val.index],
                                    y=_yoy_val.values,
                                    marker_color=["#27AE60" if v >= 0 else "#E74C3C" for v in _yoy_val.values],
                                    text=[f"{v:+.1f}%" for v in _yoy_val.values],
                                    textposition="outside",
                                ))
                                _yf2.add_hline(y=0, line_dash="dash", line_color="#bbb", line_width=1)
                                _yf2.update_layout(
                                    title=dict(text="Year-on-Year Dividend Growth (%)", font=dict(size=12)),
                                    template="plotly_white", height=320,
                                    xaxis=dict(title="Year", tickangle=-45),
                                    yaxis=dict(title="Growth (%)"),
                                    margin=dict(t=50, b=50, l=10, r=10),
                                    showlegend=False
                                )
                                st.plotly_chart(_yf2, use_container_width=True)

                        # ── Annual history table ──────────────────────────────
                        st.markdown("#### Annual Dividend History")
                        _hist_rows = []
                        for _yr in sorted(_annual_div.index, reverse=True):
                            _dv   = float(_annual_div[_yr])
                            _pv   = float(_annual_div[_yr - 1]) if (_yr - 1) in _annual_div.index else None
                            _chg  = round(_dv - _pv, 4) if _pv is not None else None
                            _cpct = round(_chg / _pv * 100, 1) if (_chg is not None and _pv and _pv > 0) else None
                            _pmts = int((_div_raw.index.year == _yr).sum())
                            _hist_rows.append({
                                "Year":            str(_yr),
                                "Annual DPS (S$)": round(_dv, 4),
                                "No. of Payments": _pmts,
                                "YoY Change (S$)": _chg,
                                "YoY Growth (%)":  _cpct,
                            })
                        _hist_df = pd.DataFrame(_hist_rows)

                        def _col_yoy(val):
                            if not isinstance(val, (int, float)) or pd.isna(val): return ''
                            if val > 0: return 'color:#1E8449;font-weight:700;'
                            if val < 0: return 'color:#C0392B;font-weight:700;'
                            return 'color:#888;'

                        st.dataframe(
                            _hist_df.style
                                .format({
                                    "Annual DPS (S$)": "S${:.4f}",
                                    "YoY Change (S$)": lambda x: f"S${x:+.4f}" if isinstance(x, float) else "—",
                                    "YoY Growth (%)":  lambda x: f"{x:+.1f}%"  if isinstance(x, float) else "—",
                                })
                                .map(_col_yoy, subset=["YoY Growth (%)", "YoY Change (S$)"]),
                            hide_index=True, use_container_width=True
                        )
                        if _fcf_note:
                            st.caption(f"FCF payout source: {_fcf_note}.")

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
      <h1 style='color:#0d2240; font-size:2rem; font-weight:700; margin:12px 0 6px 0;'>CDP Wealth Center</h1>
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
    <div style='margin-top:32px; padding:18px 22px; background:#fff8f0; border:1px solid #f5cba7;
                border-radius:10px; border-left:4px solid #e67e22;'>
      <p style='margin:0 0 6px 0; font-size:0.82rem; font-weight:700; color:#784212;'>⚠️ Important Disclaimer</p>
      <p style='margin:0; font-size:0.78rem; color:#6e4a1e; line-height:1.6;'>
        CDP Wealth Center is provided for informational and educational purposes only.
        Nothing in this application constitutes financial advice, investment advice, or a recommendation to buy, sell, or hold
        any security or financial product. All analytics, projections, and estimates are based on historical data and
        user-supplied inputs, which may be incomplete, delayed, or inaccurate.
        <strong>Past performance is not indicative of future results.</strong>
        You are solely responsible for all investment decisions you make.
        The developer of this application accepts no liability whatsoever for any financial loss, damage, or adverse outcome
        arising directly or indirectly from the use of this app.
        <strong>Always seek independent advice from a licensed financial adviser before making any investment decision.</strong>
      </p>
    </div>
    """, unsafe_allow_html=True)
