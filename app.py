import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
import warnings
import os
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import re

# Suppress warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Alpha Engine Terminal", layout="wide")
st.title("🏛️ Institutional Quantamental & Derivatives Terminal")
st.caption("Advanced data engine processing multi-factor DCF, Technicals, Greeks, Fama-French Regressions, and NLP Sentiment.")

# ==========================================
# PART 1: CORE CALCULATION ENGINES
# ==========================================

@st.cache_data(ttl=3600)
def get_live_hyperscaler_capex():
    try:
        hyperscalers = ['MSFT', 'GOOGL', 'AMZN', 'META']
        current_spend = sum([abs(yf.Ticker(t).cashflow.loc['Capital Expenditure'].iloc[0]) for t in hyperscalers])
        prior_spend = sum([abs(yf.Ticker(t).cashflow.loc['Capital Expenditure'].iloc[1]) for t in hyperscalers])
        return (current_spend - prior_spend) / prior_spend if prior_spend > 0 else 0.25
    except: return 0.25

def run_fundamental_engine(ticker_symbol, capex_growth, capture_eff):
    ticker = yf.Ticker(ticker_symbol)
    try:
        info = ticker.info
        price = info.get('currentPrice', info.get('previousClose', 1.0))
        cf = ticker.cashflow
        fcf = cf.loc['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in cf.index else 100000000
        fcf_per_share = fcf / info.get('sharesOutstanding', 1)
        gamma = 1 + (capex_growth * capture_eff)
        g_adj = 0.12 * gamma
        dcf_val = 0
        for prob, g in [(0.3, g_adj*1.4), (0.5, g_adj), (0.2, 0.12*0.5)]:
            cf_sum = sum((fcf_per_share * (1+g)**t) / (1.095)**t for t in range(1, 6))
            tv = ((fcf_per_share * (1+g)**5 * 1.03) / (0.095 - 0.03)) / (1.095)**5
            dcf_val += prob * (cf_sum + tv)
        return {"price": price, "gamma": gamma, "growth": g_adj, "value": dcf_val, "mos": (dcf_val-price)/price * 100}
    except: return None

def get_advanced_ta(ticker_symbol):
    try:
        hist = yf.Ticker(ticker_symbol).history(period="6mo")
        close = hist['Close']
        hist['SMA_20'] = close.rolling(20).mean()
        std = close.rolling(20).std()
        hist['Upper_Band'] = hist['SMA_20'] + (std * 2)
        hist['Lower_Band'] = hist['SMA_20'] - (std * 2)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        hist['MACD'] = ema_12 - ema_26
        hist['Signal_Line'] = hist['MACD'].ewm(span=9, adjust=False).mean()
        return hist
    except: return None

def get_options_chain(ticker_symbol, current_price):
    try:
        ticker = yf.Ticker(ticker_symbol)
        nearest_date = ticker.options[0]
        chain = ticker.option_chain(nearest_date)
        calls = chain.calls[(chain.calls['strike'] > current_price * 0.9) & (chain.calls['strike'] < current_price * 1.1)].copy()
        return nearest_date, calls
    except: return None, None

def run_fama_french(ticker_symbol):
    if not os.path.exists('ff_factors.csv'):
        return "Error: ff_factors.csv file not found in the current directory."
    try:
        ff = pd.read_csv('ff_factors.csv', skiprows=3, index_col=0)
        ff.index = ff.index.astype(str).str.strip()
        ff = ff[ff.index.str.len() == 6] 
        ff.index = pd.to_datetime(ff.index, format='%Y%m').to_period('M')
        ff = ff.apply(pd.to_numeric, errors='coerce').dropna()
        
        stock = yf.download(ticker_symbol, start='2019-01-01', interval='1mo', progress=False)['Close'].squeeze()
        ret = stock.pct_change().dropna() * 100
        ret.index = ret.index.to_period('M')
        
        merged = ff.join(ret.rename('Target_Return'), how='inner').dropna()
        merged['Excess_Return'] = merged['Target_Return'] - merged['RF']
        X = sm.add_constant(merged[['Mkt-RF', 'SMB', 'HML']])
        model = sm.OLS(merged['Excess_Return'], X).fit()
        return {'Alpha': model.params['const'], 'Beta': model.params['Mkt-RF'], 'SMB': model.params['SMB'], 'HML': model.params['HML'], 'R2': model.rsquared}
    except Exception as e:
        return f"Error executing regression: {e}"

def run_monte_carlo(ticker_symbol, target_days, simulations=1000):
    try:
        hist = yf.Ticker(ticker_symbol).history(period="1y")
        returns = hist['Close'].pct_change().dropna()
        last_price = hist['Close'].iloc[-1]
        mu, vol = returns.mean(), returns.std()  
        sim_data = {x: [last_price] for x in range(simulations)}
        for x in range(simulations):
            for _ in range(target_days): sim_data[x].append(sim_data[x][-1] * (np.random.normal(mu, vol) + 1))
        sim_df = pd.DataFrame(sim_data)
        final_prices = sim_df.iloc[-1]
        percentiles = {"95th": np.percentile(final_prices, 95), "50th": np.percentile(final_prices, 50), "5th": np.percentile(final_prices, 5)}
        return sim_df, percentiles
    except: return None, None

def run_nlp_sentiment(ticker_symbol):
    """Scrapes corporate text (10-K summary and recent filings/news) to measure Management Conviction vs Uncertainty."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Pull the official business summary (sourced directly from 10-K filings) and recent news
        corpus = ticker.info.get('longBusinessSummary', '')
        news = ticker.news
        if news:
            for item in news: corpus += " " + item.get('title', '') + " " + item.get('summary', '')
            
        if not corpus: return None
        
        # Base NLP Sentiment
        blob = TextBlob(corpus)
        polarity = blob.sentiment.polarity
        
        # Institutional "Lie Detector" Logic
        # We count high-conviction words vs hedging/uncertainty words
        uncertainty_words = len(re.findall(r'\b(if|may|might|subject to|risk|uncertain|volatile|headwinds)\b', corpus.lower()))
        conviction_words = len(re.findall(r'\b(strong|accelerate|expand|growth|confident|robust|surge|scale)\b', corpus.lower()))
        
        if conviction_words > (uncertainty_words * 1.5): status = "🟢 High Conviction"
        elif uncertainty_words > conviction_words: status = "🔴 High Uncertainty / Hedging"
        else: status = "⚪ Neutral Language"
        
        return {
            "polarity": polarity,
            "conviction": conviction_words,
            "uncertainty": uncertainty_words,
            "status": status,
            "word_count": len(corpus.split())
        }
    except Exception as e:
        return None

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
        
        if metrics and ta_data is not None:
            # We now have 6 Tabs!
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

            with tab4:
                st.subheader("Near-the-Money Derivative Analysis Layer")
                _, opt_chain = get_options_chain(target, metrics['price'])
                if opt_chain is not None:
                    st.dataframe(opt_chain, width="stretch")

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
                    st.error("Failed to parse NLP sentiment data for this ticker.")