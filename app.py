import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
import warnings
import os
import requests
import re
import time

# Suppress warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Alpha Engine Terminal", layout="wide")
st.title("🏛️ Institutional Quantamental & Derivatives Terminal")
st.caption("Advanced data engine processing multi-factor DCF, Technicals, Greeks, Fama-French Regressions, and NLP Sentiment.")

# ==========================================
# PART 1: CORE CALCULATION ENGINES (FMP API)
# ==========================================
import requests
import time

# Securely pull the API key from Streamlit's Vault
try:
    API_KEY = st.secrets["FMP_API_KEY"]
except:
    st.error("🔴 API Key missing. Please add FMP_API_KEY to Streamlit Secrets.")
    API_KEY = "demo"

@st.cache_data(ttl=3600, show_spinner=False)
def get_live_hyperscaler_capex():
    try:
        hyperscalers = ['MSFT', 'GOOGL', 'AMZN', 'META']
        current_spend, prior_spend = 0, 0
        for t in hyperscalers:
            url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{t}?limit=2&apikey={API_KEY}"
            res = requests.get(url).json()
            if len(res) >= 2:
                current_spend += abs(res[0].get('capitalExpenditure', 0))
                prior_spend += abs(res[1].get('capitalExpenditure', 0))
        return (current_spend - prior_spend) / prior_spend if prior_spend > 0 else 0.25
    except: return 0.25

@st.cache_data(ttl=3600, show_spinner=False)
def run_fundamental_engine(ticker_symbol, capex_growth, capture_eff):
    try:
        # Pull Quote and Cash Flow
        q_url = f"https://financialmodelingprep.com/api/v3/quote/{ticker_symbol}?apikey={API_KEY}"
        cf_url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker_symbol}?limit=1&apikey={API_KEY}"
        
        q_data = requests.get(q_url).json()[0]
        cf_data = requests.get(cf_url).json()[0]
        
        price = q_data.get('price', 1.0)
        shares = q_data.get('sharesOutstanding', 1)
        fcf = cf_data.get('freeCashFlow', 100000000)
        
        fcf_per_share = fcf / shares if shares > 0 else fcf
        gamma = 1 + (capex_growth * capture_eff)
        g_adj = 0.12 * gamma
        dcf_val = 0
        
        for prob, g in [(0.3, g_adj*1.4), (0.5, g_adj), (0.2, 0.12*0.5)]:
            cf_sum = sum((fcf_per_share * (1+g)**t) / (1.095)**t for t in range(1, 6))
            tv = ((fcf_per_share * (1+g)**5 * 1.03) / (0.095 - 0.03)) / (1.095)**5
            dcf_val += prob * (cf_sum + tv)
            
        return {"price": price, "gamma": gamma, "growth": g_adj, "value": dcf_val, "mos": (dcf_val-price)/price * 100}
    except Exception as e: return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_advanced_ta(ticker_symbol):
    try:
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker_symbol}?timeseries=150&apikey={API_KEY}"
        res = requests.get(url).json()
        if "historical" not in res: return None
        
        df = pd.DataFrame(res["historical"])
        df['Date'] = pd.to_datetime(df['date'])
        df.set_index('Date', inplace=True)
        df.sort_index(ascending=True, inplace=True) # Oldest to newest
        
        close = df['close']
        df['SMA_20'] = close.rolling(20).mean()
        std = close.rolling(20).std()
        df['Upper_Band'] = df['SMA_20'] + (std * 2)
        df['Lower_Band'] = df['SMA_20'] - (std * 2)
        
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Rename for UI compatibility
        df.rename(columns={'close': 'Close'}, inplace=True)
        return df
    except: return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_options_chain(ticker_symbol, current_price):
    # Free tier does not support options. Gracefully return None to trigger UI warning.
    return None, None

@st.cache_data(ttl=3600, show_spinner=False)
def run_monte_carlo(ticker_symbol, target_days, simulations=1000):
    try:
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker_symbol}?timeseries=252&apikey={API_KEY}"
        res = requests.get(url).json()
        df = pd.DataFrame(res["historical"])
        df.sort_values('date', inplace=True)
        
        returns = df['close'].pct_change().dropna()
        last_price = df['close'].iloc[-1]
        mu, vol = returns.mean(), returns.std()  
        
        sim_data = {x: [last_price] for x in range(simulations)}
        for x in range(simulations):
            for _ in range(target_days): sim_data[x].append(sim_data[x][-1] * (np.random.normal(mu, vol) + 1))
            
        sim_df = pd.DataFrame(sim_data)
        final_prices = sim_df.iloc[-1]
        percentiles = {"95th": np.percentile(final_prices, 95), "50th": np.percentile(final_prices, 50), "5th": np.percentile(final_prices, 5)}
        return sim_df, percentiles
    except: return None, None

@st.cache_data(ttl=3600, show_spinner=False)
def run_fama_french(ticker_symbol):
    if not os.path.exists('ff_factors.csv'): return "Error: ff_factors.csv file not found."
    try:
        # Load Fama-French Data
        ff = pd.read_csv('ff_factors.csv', skiprows=3, index_col=0)
        ff.index = ff.index.astype(str).str.strip()
        ff = ff[ff.index.str.len() == 6] 
        ff.index = pd.to_datetime(ff.index, format='%Y%m').to_period('M')
        ff = ff.apply(pd.to_numeric, errors='coerce').dropna()
        
        # Load FMP Monthly Data
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker_symbol}?timeseries=1200&apikey={API_KEY}"
        res = requests.get(url).json()
        stock = pd.DataFrame(res["historical"])
        stock['date'] = pd.to_datetime(stock['date'])
        stock.set_index('date', inplace=True)
        stock.sort_index(ascending=True, inplace=True)
        
        # Resample to monthly and calculate returns
        monthly_stock = stock['close'].resample('M').last()
        ret = monthly_stock.pct_change().dropna() * 100
        ret.index = ret.index.to_period('M')
        
        merged = ff.join(ret.rename('Target_Return'), how='inner').dropna()
        merged['Excess_Return'] = merged['Target_Return'] - merged['RF']
        X = sm.add_constant(merged[['Mkt-RF', 'SMB', 'HML']])
        model = sm.OLS(merged['Excess_Return'], X).fit()
        return {'Alpha': model.params['const'], 'Beta': model.params['Mkt-RF'], 'SMB': model.params['SMB'], 'HML': model.params['HML'], 'R2': model.rsquared}
    except Exception as e: return f"Error executing regression: {e}"

@st.cache_data(ttl=3600, show_spinner=False)
def run_nlp_sentiment(ticker_symbol):
    try:
        prof_data = fetch_fmp(f"profile/{ticker_symbol}")
        news_data = fetch_fmp(f"stock_news?tickers={ticker_symbol}&limit=10")
        
        corpus = prof_data[0].get('description', '') if prof_data else ""
        if news_data:
            for item in news_data: corpus += " " + item.get('title', '') + " " + item.get('text', '')
            
        if not corpus: return None
        
        # Institutional "Lie Detector" Logic (Pure Regex)
        uncertainty_words = len(re.findall(r'\b(if|may|might|subject to|risk|uncertain|volatile|headwinds)\b', corpus.lower()))
        conviction_words = len(re.findall(r'\b(strong|accelerate|expand|growth|confident|robust|surge|scale)\b', corpus.lower()))
        
        if conviction_words > (uncertainty_words * 1.5): status = "🟢 High Conviction"
        elif uncertainty_words > conviction_words: status = "🔴 High Uncertainty / Hedging"
        else: status = "⚪ Neutral Language"
        
        return {
            "polarity": 0.00, # Deprecated TextBlob score
            "conviction": conviction_words,
            "uncertainty": uncertainty_words,
            "status": status,
            "word_count": len(corpus.split())
        }
    except Exception as e: return None
# ==========================================
# PART 2: TERMINAL UI (SIDEBAR)
# ==========================================
target = st.sidebar.text_input("Target Ticker:", value="VRT").upper().strip()
macro_growth = st.sidebar.slider("Hyperscaler CapEx Expansion:", 0.0, 1.0, 0.25, format="%.1f%%")
capture_eff = st.sidebar.slider("Asset Capture Efficiency Matrix:", 0.0, 1.0, 0.50)
sim_days = st.sidebar.slider("Risk Matrix Days:", 30, 365, 252)

# ==========================================
# PART 3: MAIN WORKSTATION DASHBOARD
# ==========================================
if st.sidebar.button("⚡ Execute Full Terminal Analysis"):
    with st.spinner(f"Compiling advanced arrays for {target}..."):
        metrics = run_fundamental_engine(target, macro_growth, capture_eff)
        ta_data = get_advanced_ta(target)
        
        # --- ERROR TRAPPING ---
        if metrics is None:
            st.error(f"🔴 **CRITICAL FAILURE:** Could not fetch fundamental data for {target}. Yahoo Finance may be rate-limiting the cloud server, or the ticker is invalid. Try a major ticker like 'AAPL' to test the connection.")
        
        if ta_data is None:
            st.error(f"🔴 **CRITICAL FAILURE:** Could not fetch historical price data for {target}. The technical analysis engine failed to initialize.")
            
        # --- RENDER TABS IF DATA EXISTS ---
        if metrics is not None and ta_data is not None:
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 DCF Fundamentals", 
                "📈 Technical Action", 
                "🎲 Risk Matrix", 
                "⛓️ Options Chain", 
                "🏛️ Fama-French",
                "🕵️ NLP Sentiment"
            ])
            
            with tab1:
                st.subheader("Asset Valuation & Supply Chain Mapping")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Current Spot", f"${metrics['price']:.2f}")
                c2.metric("Intrinsic Value", f"${metrics['value']:.2f}")
                c3.metric("Margin of Safety", f"{metrics['mos']:.1f}%", delta=f"{metrics['mos']:.1f}%")
                c4.metric("Growth Rate", f"{metrics['growth']*100:.1f}%")

            with tab2:
                st.subheader("Price Channels, Momentum, & Trend Indicators")
                
                last_close = ta_data['Close'].iloc[-1]
                upper = ta_data['Upper_Band'].iloc[-1]
                lower = ta_data['Lower_Band'].iloc[-1]
                rsi = ta_data['RSI'].iloc[-1]
                
                if last_close > upper: bb_status = "🟢 Breakout / Overextended"
                elif last_close < lower: bb_status = "🔴 Poor / Breakdown"
                else: bb_status = "⚪ Neutral / Ranging"
                
                if rsi > 70: rsi_status = "Overbought"
                elif rsi < 30: rsi_status = "Oversold"
                else: rsi_status = "Neutral"

                t2_c1, t2_c2 = st.columns(2)
                t2_c1.metric("Current Volatility Action", bb_status)
                t2_c2.metric("14-Day RSI Condition", f"{rsi:.1f} ({rsi_status})")
                st.divider()

                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    st.markdown("**Volatility Channels (Bollinger Bands)**")
                    st.line_chart(ta_data[['Close', 'Upper_Band', 'Lower_Band']], width="stretch")
                with col_chart2:
                    st.markdown("**Trend Velocity (MACD)**")
                    st.line_chart(ta_data[['MACD', 'Signal_Line']], width="stretch")
                st.markdown("**Structural Momentum (14-Day RSI)**")
                st.line_chart(ta_data[['RSI']], width="stretch")

            with tab3:
                st.subheader(f"Geometric Brownian Motion Modeling ({sim_days} Trading Days)")
                sim_data, percentiles = run_monte_carlo(target, sim_days)
                if sim_data is not None:
                    final_prices = sim_data.iloc[-1]
                    counts, bins = np.histogram(final_prices, bins=40)
                    hist_df = pd.DataFrame({'Probability Density': counts}, index=np.round(bins[:-1], 2).astype(str))
                    st.bar_chart(hist_df, width="stretch")
                    mc1, mc2, mc3 = st.columns(3)
                    mc3.metric("95th Pctl (Breakout)", f"${percentiles['95th']:.2f}")
                    mc2.metric("50th Pctl (Median)", f"${percentiles['50th']:.2f}")
                    mc1.metric("5th Pctl (Drawdown)", f"${percentiles['5th']:.2f}")
                    with st.expander("👁️ View Raw Simulation Paths (Spaghetti Chart)"):
                        st.line_chart(sim_data.iloc[:, :100], width="stretch")
                else:
                    st.warning("Monte Carlo simulation failed to run.")

            with tab4:
                st.subheader("Near-the-Money Derivative Analysis Layer")
                _, opt_chain = get_options_chain(target, metrics['price'])
                if opt_chain is not None:
                    st.dataframe(opt_chain, width="stretch")
                else:
                    st.warning("Options chain data unavailable for this ticker right now.")

            with tab5:
                st.subheader("Fama-French 3-Factor Asset Pricing Regression")
                res = run_fama_french(target)
                if isinstance(res, dict):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Alpha (Unique Edge)", f"{res['Alpha']:.2f}%")
                    c2.metric("Market Beta (Mkt-RF)", f"{res['Beta']:.2f}")
                    c3.metric("Size Premium (SMB)", f"{res['SMB']:.2f}")
                    c4.metric("Value Premium (HML)", f"{res['HML']:.2f}")
                    st.info(f"**Regression R-Squared:** {res['R2']*100:.1f}%")
                else:
                    st.error(res)
                    
            with tab6:
                st.subheader("NLP Management Sentiment & 'Lie Detector'")
                nlp_res = run_nlp_sentiment(target)
                if nlp_res:
                    st.markdown(f"**Corpus Scanned:** ~{nlp_res['word_count']} words (10-K Business Summaries & Recent Filings)")
                    st.metric("Overall Management Posture", nlp_res['status'])
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Conviction Words", nlp_res['conviction'])
                    c2.metric("Uncertainty/Hedging Words", nlp_res['uncertainty'])
                    c3.metric("Raw Polarity Score", f"{nlp_res['polarity']:.2f}")
                    st.divider()
                    st.info("""
                    **How to read this:**
                    * **Conviction Words:** Tracks frequency of strong, forward-looking language (e.g., "accelerate", "expand", "scale"). 
                    * **Uncertainty Words:** Tracks frequency of legal hedging and risk language (e.g., "subject to", "volatile", "might").
                    * A high ratio of Conviction to Uncertainty often precedes massive CapEx spending or earnings beats.
                    """)
                else:
                    st.warning("Failed to parse NLP sentiment data. TextBlob may be missing dictionaries or no news was found.")
