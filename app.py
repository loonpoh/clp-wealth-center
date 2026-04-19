import streamlit as st
import pandas as pd
import pdfplumber
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import re
import numpy as np
from datetime import datetime, timedelta

# --- 1. SYSTEM CONFIG & STANDARDIZATION ---
st.set_page_config(layout="wide", page_title="CLP Wealth Center", page_icon="🏦")

# --- 2. ASSET REGISTRY ---
MASTER_INTEL = {
    "DBS": {"Rate": 3.24, "PB": 1.55, "Ticker": "D05.SI", "Sector": "Financial Services"},
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
    "HPH TRUST USD": {"Rate": 0.02, "PB": 0.35, "Ticker": "NS8U.SI", "Sector": "REITs & Business Trusts"},
    "ASTREAVIB310318": {"Rate": 0.03, "PB": 1.00, "Ticker": None, "Sector": "Fixed Income"},
    "SBDEC17 GX17120W": {"Rate": 0.03, "PB": 1.00, "Ticker": None, "Sector": "Fixed Income"}
}

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
    name_up = str(raw_name).upper()
    if name_up in MASTER_INTEL:
        return MASTER_INTEL[name_up], name_up
        
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
            return MASTER_INTEL[clean_name], clean_name
            
    return {"Rate": 0.0, "PB": 1.0, "Ticker": None, "Sector": guess_sector(name_up)}, name_up

def extract_pdf(file):
    data = []
    seen = set()
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for row in table:
                    row = [str(c).replace('\n', ' ').strip() for c in row if c is not None and str(c).strip() != '']
                    if len(row) < 3: continue
                    
                    raw_name = row[0].upper()
                    
                    ignore_keywords = ["SECURITY", "BALANCE", "DATE", "TOTAL", "PAGE", "STATEMENT", "PORTFOLIO", "SUB-TOTAL"]
                    if any(x in raw_name for x in ignore_keywords): 
                        continue
                    
                    if re.match(r'^\d{1,2}[\s\-/]+(?:[A-Z]{3}|\d{1,2})[\s\-/]+\d{2,4}', raw_name):
                        continue
                    
                    try:
                        numbers = [clean_val(c) for c in row[1:] if re.search(r'\d', str(c))]
                        if len(numbers) >= 2:
                            qty = numbers[0]
                            mkt = numbers[-1]
                            
                            if qty < 1 or mkt <= 0: continue
                            
                            intel, clean_name = resolve_intel(raw_name)
                            
                            if clean_name in seen: continue
                                
                            data.append({
                                "Security": clean_name, 
                                "Sector": intel.get("Sector", guess_sector(clean_name)), 
                                "Quantity": qty, 
                                "AUM (SGD)": mkt, 
                                "DPS": intel["Rate"], 
                                "P/B Ratio": intel["PB"], 
                                "Ticker": intel["Ticker"]
                            })
                            seen.add(clean_name)
                    except Exception as e:
                        continue
    return pd.DataFrame(data)

def color_returns(val):
    if isinstance(val, (int, float)):
        if val > 0: return 'background-color: rgba(39, 174, 96, 0.15); color: #1E8449; font-weight: bold;'
        elif val < 0: return 'background-color: rgba(214, 48, 49, 0.15); color: #C0392B; font-weight: bold;'
    return ''

# --- 3. DASHBOARD UI ---
st.sidebar.header("🏦 CLP COMMAND CENTER")
uploaded_file = st.sidebar.file_uploader("Step 1: Upload Portfolio PDF", type="pdf")

with st.sidebar.expander("⚙️ App Maintenance Guide"):
    st.markdown("""
    **The Golden Rule of New Stocks**
    
    If you purchase a brand new asset that you have never owned before, the app will automatically calculate its AUM, but it will *hide* it from the Market Benchmark and Audit tabs to prevent crashes.
    
    **To unlock full auditing for a new stock:**
    1. Open `app.py` in your GitHub repository.
    2. Locate the `MASTER_INTEL` list near the top of the code.
    3. Add your new stock name and its `.SI` ticker symbol to the list. 
    *(Example: `"PANUNITED": {"Rate": 0.0, "PB": 1.0, "Ticker": "P52.SI", "Sector": "Industrials"}`)*
    4. Click **Commit changes**. Streamlit will instantly refresh your live app!
    """)
st.sidebar.markdown("---")

t1, t2, t3, t4 = st.tabs(["📊 Asset Discovery", "📈 Market Benchmark", "📂 Verification Hub", "📜 Strategic Advisory"])

if uploaded_file:
    df_raw = extract_pdf(uploaded_file)
    if not df_raw.empty:
        all_sects = sorted(df_raw['Sector'].unique())
        sel_sects = st.sidebar.multiselect("Filter Analysis Universe", all_sects, default=all_sects)
        df = df_raw[df_raw['Sector'].isin(sel_sects)].copy()
        
        # --- ENHANCED METRICS & LABEL ENGINE ---
        total_aum = df['AUM (SGD)'].sum()
        
        df["Annual Dividend (SGD)"] = (df["Quantity"] * df["DPS"]).round(2)
        total_inc = df['Annual Dividend (SGD)'].sum()
        
        df["Dividend Yield (%)"] = (df["Annual Dividend (SGD)"] / df["AUM (SGD)"] * 100).round(2)
        
        # Calculate strict mathematical weights
        df["%Portfolio"] = (df["AUM (SGD)"] / total_aum * 100).round(2) if total_aum > 0 else 0
        df["%Income"] = (df["Annual Dividend (SGD)"] / total_inc * 100).round(2) if total_inc > 0 else 0
        
        # Bake the weights directly into the display labels using 1 decimal place for neatness
        df["Alloc_Label"] = df.apply(lambda r: f"{r['Security']} ({r['%Portfolio']:.1f}%)", axis=1)
        df["Inc_Label"] = df.apply(lambda r: f"{r['Security']} ({r['%Income']:.1f}%)", axis=1)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total AUM", f"S${total_aum:,.2f}")
        m2.metric("Annualized Income", f"S${total_inc:,.2f}")
        m3.metric("Portfolio Yield", f"{(total_inc/total_aum*100):.2f}%" if total_aum > 0 else "0.00%")

        with t1:
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("<h4 style='text-align: center;'>Capital Allocation</h4>", unsafe_allow_html=True)
                # Map the Treemap to use the new 'Alloc_Label'
                fig_tree = px.treemap(df, path=['Sector', 'Alloc_Label'], values='AUM (SGD)', color='Sector')
                fig_tree.update_layout(margin=dict(t=10, l=10, r=10, b=10)) 
                st.plotly_chart(fig_tree, use_container_width=True)
                
            with c2:
                st.markdown("<h4 style='text-align: center;'>Income Stream</h4>", unsafe_allow_html=True)
                # Map the Sunburst to use the new 'Inc_Label'
                fig_sun = px.sunburst(df, path=['Sector', 'Inc_Label'], values='Annual Dividend (SGD)', color='Sector')
                fig_sun.update_layout(margin=dict(t=10, l=10, r=10, b=10))
                st.plotly_chart(fig_sun, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h4 style='text-align: center;'>Asset Efficiency Matrix: Valuation vs. Yield</h4>", unsafe_allow_html=True)
            fig_scatter = px.scatter(df[df['Sector']!='Fixed Income'], x="P/B Ratio", y="Dividend Yield (%)", size="AUM (SGD)", color="Sector", hover_name="Security", template="plotly_dark", height=500)
            fig_scatter.update_layout(margin=dict(t=20, l=10, r=10, b=10))
            st.plotly_chart(fig_scatter, use_container_width=True)

        with t2:
            st.markdown("### 📈 Total Return vs STI Benchmark")
            eligible = df[df['Ticker'].notnull()]['Security'].unique()
            
            if len(eligible) > 0:
                c_s, c_i = st.columns(2)
                target = c_s.selectbox("Select Asset", options=eligible)
                yrs = c_i.selectbox("Horizon", [1, 5, 10, 15, 20], format_func=lambda x: f"{x} Years")
                
                if st.button("Launch Comparison"):
                    tick = df[df['Security'] == target]['Ticker'].iloc[0] 
                    h_a = yf.download(tick, start=(datetime.now()-timedelta(days=yrs*365)), end=datetime.now(), auto_adjust=True, progress=False)
                    h_s = yf.download("^STI", start=(datetime.now()-timedelta(days=yrs*365)), end=datetime.now(), auto_adjust=True, progress=False)
                    
                    if not h_a.empty and not h_s.empty:
                        c_a = h_a['Close'].iloc[:, 0] if isinstance(h_a['Close'], pd.DataFrame) else h_a['Close']
                        c_s = h_s['Close'].iloc[:, 0] if isinstance(h_s['Close'], pd.DataFrame) else h_s['Close']
                        asset_norm = (c_a / float(c_a.iloc[0])) * 100
                        sti_norm = (c_s / float(c_s.iloc[0])) * 100
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=asset_norm.index, y=asset_norm, name=target, line=dict(color='#27AE60', width=3)))
                        fig.add_trace(go.Scatter(x=sti_norm.index, y=sti_norm, name="STI Benchmark", line=dict(color='#95A5A6', dash='dot')))
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No benchmarkable assets found in the current selection.")

        with t3:
            for s in sorted(df['Sector'].unique()):
                sdf = df[df['Sector']==s].copy().sort_values('AUM (SGD)', ascending=False)
                sdf.insert(0, 'S/N', range(1, len(sdf)+1))
                with st.expander(f"{s.upper()} VERIFICATION PANEL"):
                    col_chart, col_table = st.columns([1, 1])
                    with col_chart:
                        st.plotly_chart(px.bar(sdf, x='Security', y='AUM (SGD)', color='Dividend Yield (%)', color_continuous_scale='Blues'), use_container_width=True)
                    with col_table:
                        st.dataframe(sdf[['S/N','Security','Quantity','DPS','Annual Dividend (SGD)','Dividend Yield (%)','%Portfolio','AUM (SGD)']].style.format({
                            'DPS': '{:.2f}', 'Annual Dividend (SGD)': '{:,.2f}', 'Dividend Yield (%)': '{:.2f}%', '%Portfolio': '{:.2f}%', 'AUM (SGD)': '{:,.2f}'
                        }).background_gradient(subset=['Dividend Yield (%)'], cmap='Blues', vmin=0), hide_index=True, use_container_width=True)

        with t4:
            st.markdown("## 📜 Strategic Advisory & Health Audit")
            
            @st.cache_data(ttl=3600)
            def scan_p(_data):
                m = {}
                for _, row in _data[_data['Ticker'].notnull()].iterrows():
                    try:
                        h = yf.download(row['Ticker'], period="5y", auto_adjust=True, progress=False)
                        if not h.empty:
                            c = h['Close'].iloc[:, 0] if isinstance(h['Close'], pd.DataFrame) else h['Close']
                            m[row['Security']] = float(c.iloc[-1]) / float(c.iloc[0]) - 1.0
                    except: continue
                return m
            
            p_map = scan_p(df)
            losers = [k for k, v in p_map.items() if float(v) < 0.0]
            winners = sorted([k for k, v in p_map.items() if float(v) >= 0.0], key=lambda x: float(p_map[x]), reverse=True)[:10]

            col_a, col_b = st.columns(2)
            health = 100 - (len(losers) * 5)
            with col_a:
                st.metric("Portfolio Health Score", f"{max(health, 0)}/100")
                top_divs = df.nlargest(2, 'Annual Dividend (SGD)')['Security'].tolist()
                anchor_text = f"**{', '.join(top_divs)}**" if top_divs else "diversified cash holdings"
                st.markdown(f"**Strengths:** Core stability in {anchor_text}.")
            with col_b:
                st.markdown(f"**Priority:** Investigate **{len(losers)}** assets underperforming capital benchmarks.")

            st.markdown("---")
            
            c_aud1, c_aud2 = st.columns(2)
            with c_aud1:
                st.markdown("### 💎 Wealth Creator Audit")
                if len(winners) > 0:
                    sel_w = st.multiselect("High-Performers", options=winners, default=winners[:2])
                    wy = st.multiselect("Creator Horizons", options=[5, 10, 15, 20], default=[5, 10], key="wy")
                    if st.button("📈 Audit Creators"):
                        res_w = []
                        for n in sel_w:
                            rd = {"Security": n}
                            for y in wy:
                                tick = df[df['Security'] == n]['Ticker'].iloc[0]
                                h = yf.download(tick, start=(datetime.now()-timedelta(days=y*365)), end=datetime.now(), auto_adjust=True, progress=False)
                                if not h.empty:
                                    c = h['Close'].iloc[:, 0] if isinstance(h['Close'], pd.DataFrame) else h['Close']
                                    rd[f"{y}Y Total Return"] = (float(c.iloc[-1]) / float(c.iloc[0]) - 1.0) * 100
                            res_w.append(rd)
                        df_w = pd.DataFrame(res_w)
                        st.dataframe(df_w.style.format({c: "{:.2f}%" for c in df_w.columns if "Return" in c}, na_rep="-").map(color_returns), use_container_width=True, hide_index=True)
                else:
                    st.info("No positive returning assets found in the current selection.")

            with c_aud2:
                st.markdown("### 🚨 Wealth Destroyer Audit")
                if len(losers) > 0:
                    sel_l = st.multiselect("Suspects", options=losers)
                    ly = st.multiselect("Destroyer Horizons", options=[5, 10, 15, 20], default=[5, 10], key="ly")
                    if st.button("🚀 Audit Destroyers"):
                        res_l = []
                        for n in sel_l:
                            rd = {"Security": n}
                            for y in ly:
                                tick = df[df['Security'] == n]['Ticker'].iloc[0]
                                h = yf.download(tick, start=(datetime.now()-timedelta(days=y*365)), end=datetime.now(), auto_adjust=True, progress=False)
                                if not h.empty:
                                    c = h['Close'].iloc[:, 0] if isinstance(h['Close'], pd.DataFrame) else h['Close']
                                    rd[f"{y}Y Total Return"] = (float(c.iloc[-1]) / float(c.iloc[0]) - 1.0) * 100
                            res_l.append(rd)
                        df_l = pd.DataFrame(res_l)
                        st.dataframe(df_l.style.format({c: "{:.2f}%" for c in df_l.columns if "Return" in c}, na_rep="-").map(color_returns), use_container_width=True, hide_index=True)
                else:
                    st.success("No wealth destroying assets detected!")

            st.caption("*High Performers and Suspects are selected based on their last 5-Year Total Return.*")

            st.markdown("---")

            st.markdown("### ⚖️ Portfolio Rebalance Simulator")
            col_sell, col_buy = st.columns(2)
            
            with col_sell:
                st.error("📉 Source of Funds (Sell Target)")
                held_losers = [l for l in losers if l in df['Security'].values]
                sim_sell = st.multiselect("Select assets to liquidate", options=df['Security'].unique(), default=held_losers)
            
            with col_buy:
                st.success("📈 Destination of Funds (Buy Target)")
                sim_buy = st.multiselect("Select assets to acquire", options=df['Security'].unique(), default=winners[:2] if len(winners) > 1 else None)

            if st.button("🔄 Execute Simulation"):
                if not sim_sell or not sim_buy:
                    st.warning("⚠️ Please select at least one asset to sell and one to buy to run the simulation.")
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
                                    live_price = float(c.iloc[-1])
                                    live_yield = dps / live_price
                                    new_income_added += (capital_per_buy * live_yield)
                                else:
                                    new_income_added += (capital_per_buy * 0.04)
                            except:
                                new_income_added += (capital_per_buy * 0.04)

                    new_aum = curr_aum 
                    new_inc = curr_inc - lost_income + new_income_added
                    new_yield = (new_inc / new_aum) * 100

                    st.success(f"**Simulation Complete:** S${freed_capital:,.2f} recycled from {len(sim_sell)} assets into {len(sim_buy)} assets.")
                    
                    st.markdown("""<style>[data-testid="stMetricDelta"] svg {display: none;}</style>""", unsafe_allow_html=True)
                    r1, r2, r3 = st.columns(3)
                    
                    r1.metric("Projected AUM (Excl. Fees)", f"S${new_aum:,.2f}", "S$0.00 (Neutral)")
                    r2.metric("Projected Annual Income", f"S${new_inc:,.2f}", f"S${new_inc - curr_inc:,.2f} per year")
                    r3.metric("Projected Portfolio Yield", f"{new_yield:.2f}%", f"{new_yield - curr_yield:.2f}% shift")

else:
    st.info("👋 Welcome. Please upload your CDP PDF or similar portfolio statement to begin.")