# Main Streamlit application for the Financial Analytics Web App
#app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import traceback

# Local Imports
from binomial_model import BinomialModel
from newton_raphson import ImpliedVolatilitySolver
from live_data import FinancialDataFetcher
from finance_bot import FinanceAIBot
import config

# Set Streamlit page configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="📈",
    layout="wide"
)

# Initialize resources with caching
@st.cache_resource
def initialize_resources():
    binomial_model = BinomialModel()
    iv_solver = ImpliedVolatilitySolver()
    data_fetcher = FinancialDataFetcher()
    finance_bot = FinanceAIBot()
    return binomial_model, iv_solver, data_fetcher, finance_bot

binomial_model, iv_solver, data_fetcher, finance_bot = initialize_resources()

def calculate_years_to_expiry(expiry_date):
    now = datetime.now()
    days = (expiry_date - now).days + (expiry_date - now).seconds / (24 * 3600)
    return max(0, days / 365.0)

# App title and description
st.title("📉 Financial Analytics AI Platform")
st.markdown(config.APP_DESCRIPTION)

# Stock Assistant Floating Button
def show_stock_assistant():
    with st.expander("🤖 Stock Assistant", expanded=False):
        st.markdown("""
        *How can I help you today?*  
        Select an option below to get insights about your stock or option.
        """)
        
        analysis_type = st.selectbox(
            "Choose analysis type",
            ["Find Risk Level", "Best Time to Buy/Sell", "Generate Graph", "Detailed Analysis"]
        )
        
        if analysis_type == "Find Risk Level":
            st.markdown("### Risk Assessment")
            if 'risk_assessment' in st.session_state:
                risk = st.session_state.risk_assessment
                if risk['level'] == "High":
                    st.error(f"Risk Level: {risk['level']}")
                elif risk['level'] in ["Medium-High", "Medium"]:
                    st.warning(f"Risk Level: {risk['level']}")
                else:
                    st.success(f"Risk Level: {risk['level']}")
                
                st.write("*Reasons:*")
                for reason in risk['reasons']:
                    st.write(f"- {reason}")
            else:
                st.info("Run a calculation first to see risk assessment")
        
        elif analysis_type == "Best Time to Buy/Sell":
            st.markdown("### Optimal Trading Strategy")
            if 'risk_assessment' in st.session_state:
                risk = st.session_state.risk_assessment
                st.write(f"*Recommendation:* {risk['recommendation']}")
                
                # Calculate best time based on volatility and price trends
                if 'hist_vol' in st.session_state:
                    hist_vol = st.session_state.hist_vol
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    
                    if risk['level'] == "Low" and "buy" in risk['recommendation'].lower():
                        st.success(f"Best time to buy: Now ({current_date})")
                        st.write("Low risk with favorable conditions for buying")
                    elif risk['level'] == "High" and "sell" in risk['recommendation'].lower():
                        st.warning(f"Best time to sell: Now ({current_date})")
                        st.write("High risk suggests selling may be optimal")
                    else:
                        st.info("Current conditions suggest holding position")
            else:
                st.info("Run a calculation first to get trading recommendations")
        
        elif analysis_type == "Generate Graph":
            st.markdown("### Visualization Options")
            graph_type = st.selectbox(
                "Select graph type",
                ["Price vs Volatility", "Price vs Stock Price", "Historical Volatility Trend"]
            )
            
            if graph_type == "Price vs Volatility" and 'vol_df' in st.session_state:
                fig = px.line(st.session_state.vol_df, x='Volatility', y='Option Price', 
                             title='Option Price vs. Volatility')
                fig.update_layout(xaxis_title='Volatility', yaxis_title='Option Price ($)')
                st.plotly_chart(fig)
            
            elif graph_type == "Price vs Stock Price" and 'stock_df' in st.session_state:
                fig = px.line(st.session_state.stock_df, x='Stock Price', y='Option Price', 
                             title='Option Price vs. Stock Price')
                fig.update_layout(xaxis_title='Stock Price ($)', yaxis_title='Option Price ($)')
                fig.add_vline(x=st.session_state.strike_price, line_dash="dash", 
                             line_color="red", annotation_text=f"Strike: ${st.session_state.strike_price}")
                st.plotly_chart(fig)
            
            elif graph_type == "Historical Volatility Trend" and 'hist_data' in st.session_state:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.hist_data.index,
                    y=st.session_state.hist_data['Close'],
                    mode='lines',
                    name='Price'
                ))
                fig.add_trace(go.Scatter(
                    x=st.session_state.hist_data.index,
                    y=st.session_state.hist_data['Bollinger_Upper'],
                    mode='lines',
                    name='Upper Bollinger Band',
                    line=dict(color='red')
                ))
                fig.add_trace(go.Scatter(
                    x=st.session_state.hist_data.index,
                    y=st.session_state.hist_data['Bollinger_Lower'],
                    mode='lines',
                    name='Lower Bollinger Band',
                    line=dict(color='green')
                ))
                fig.update_layout(
                    title='Price with Bollinger Bands',
                    xaxis_title='Date',
                    yaxis_title='Price ($)'
                )
                st.plotly_chart(fig)
            else:
                st.info("Generate some data first to view graphs")
        
        elif analysis_type == "Detailed Analysis":
            if 'ticker' in st.session_state and 'hist_data' in st.session_state:
                ticker = st.session_state.ticker
                hist_data = st.session_state.hist_data
                
                st.markdown("### Technical Analysis Summary")
                
                # Calculate metrics
                last_close = hist_data['Close'].iloc[-1]
                ma20 = hist_data['Moving_Avg_20'].iloc[-1]
                ma50 = hist_data['Moving_Avg_50'].iloc[-1]
                rsi = hist_data['RSI'].iloc[-1] if 'RSI' in hist_data.columns else None
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${last_close:.2f}")
                    st.metric("20-Day MA", f"${ma20:.2f}", 
                             delta=f"{(last_close - ma20)/ma20:.2%}")
                
                with col2:
                    st.metric("50-Day MA", f"${ma50:.2f}", 
                             delta=f"{(last_close - ma50)/ma50:.2%}")
                    if rsi:
                        st.metric("RSI", f"{rsi:.1f}", 
                                 delta="Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral")
                
                with col3:
                    volatility = hist_data['Daily_Return'].std() * np.sqrt(252)
                    st.metric("Annualized Volatility", f"{volatility:.2%}")
                    
                    # Volume analysis
                    avg_volume = hist_data['Volume'].mean()
                    last_volume = hist_data['Volume'].iloc[-1]
                    st.metric("Volume", f"{last_volume/1e6:.1f}M", 
                              delta=f"{last_volume/avg_volume:.1f}x avg")
                
                # Trend analysis
                st.markdown("#### Trend Analysis")
                if last_close > ma20 > ma50:
                    st.success("Strong uptrend (Price > 20MA > 50MA)")
                elif last_close < ma20 < ma50:
                    st.error("Strong downtrend (Price < 20MA < 50MA)")
                else:
                    st.info("Mixed or sideways trend")
            else:
                st.info("Fetch market data first to get detailed analysis")

# Show the assistant in sidebar
with st.sidebar:
    show_stock_assistant()

# Tabs
opt_tab, data_tab, bot_tab = st.tabs(["Option Pricing & IV", "Market Data", "AI Finance Assistant"])

# TAB 1: Option Pricing & Implied Volatility
with opt_tab:
    st.header("Option Pricing & Implied Volatility Calculator")
    col1, col2 = st.columns(2)

    with col1:
        ticker = st.text_input("Stock Ticker", "AAPL", key="option_ticker")
        st.session_state.ticker = ticker
        
        try:
            current_price = data_fetcher.get_stock_price(ticker)
            st.success(f"Current price: ${current_price:.2f}")
        except Exception as e:
            current_price = 100.0
            st.warning(f"Using default price: $100.00 (Error: {str(e)})")

        stock_price = st.number_input("Stock Price", value=current_price, step=0.01)
        strike_price = st.number_input("Strike Price", value=current_price, step=0.01)
        st.session_state.strike_price = strike_price

        today = datetime.now().date()
        expiry_date = st.date_input("Expiry Date", today + timedelta(days=30))
        time_to_expiry = calculate_years_to_expiry(datetime.combine(expiry_date, datetime.min.time()))
        st.write(f"Time to expiry: {time_to_expiry:.6f} years")

        try:
            risk_free_rate = data_fetcher.get_risk_free_rate()
        except:
            risk_free_rate = config.DEFAULT_RISK_FREE_RATE

        interest_rate = st.number_input("Risk-Free Interest Rate", value=risk_free_rate, step=0.001, format="%.4f")

    with col2:
        option_type = st.selectbox("Option Type", ["call", "put"])
        try:
            hist_vol = data_fetcher.calculate_historical_volatility(ticker)
            st.session_state.hist_vol = hist_vol
            st.success(f"Historical volatility: {hist_vol:.4f}")
        except:
            hist_vol = 0.3
        volatility = st.number_input("Volatility", value=hist_vol, step=0.01, format="%.4f")
        steps = st.slider("Steps", 10, 1000, 50, step=10)
        american = st.checkbox("American Option", False)
        calculation_mode = st.radio("Calculation Mode", ["Price Option", "Find Implied Volatility"])
        market_price = st.number_input("Market Price", value=0.0 if calculation_mode=="Price Option" else 5.0, step=0.01)

    if st.button("Calculate Option"):
        with st.spinner("Calculating..."):
            try:
                if calculation_mode == "Price Option":
                    price = binomial_model.price_option(stock_price, strike_price, time_to_expiry, interest_rate, 
                                                      volatility, steps, option_type, american)
                    greeks = binomial_model.calculate_option_greeks(stock_price, strike_price, time_to_expiry, 
                                                                   interest_rate, volatility, steps, option_type, american)

                    st.subheader("Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Option Price", f"${price:.4f}")
                        st.metric("Delta", f"{greeks['delta']:.4f}")
                        st.metric("Gamma", f"{greeks['gamma']:.4f}")
                    
                    with col2:
                        st.metric("Theta", f"{greeks['theta']:.4f}")
                        st.metric("Vega", f"{greeks['vega']:.4f}")
                        st.metric("Rho", f"{greeks['rho']:.4f}")
                    
                    # Store risk assessment for assistant
                    st.session_state.risk_assessment = greeks['risk_assessment']
                    
                    # Generate price vs. volatility chart data
                    vol_range = np.linspace(max(0.05, volatility - 0.2), volatility + 0.2, 50)
                    prices = []
                    for vol in vol_range:
                        p = binomial_model.price_option(stock_price, strike_price, time_to_expiry, interest_rate, 
                                                      vol, steps, option_type, american)
                        prices.append(p)
                    
                    st.session_state.vol_df = pd.DataFrame({
                        'Volatility': vol_range,
                        'Option Price': prices
                    })
                    
                    # Generate price vs. stock price chart data
                    stock_range = np.linspace(stock_price * 0.7, stock_price * 1.3, 50)
                    prices = []
                    for s in stock_range:
                        p = binomial_model.price_option(s, strike_price, time_to_expiry, interest_rate, 
                                                      volatility, steps, option_type, american)
                        prices.append(p)
                    
                    st.session_state.stock_df = pd.DataFrame({
                        'Stock Price': stock_range,
                        'Option Price': prices
                    })
                
                else:  # Find Implied Volatility
                    if market_price <= 0:
                        st.error("Market price must be positive for implied volatility calculation")
                    else:
                        result = iv_solver.calculate_implied_volatility(stock_price, strike_price, time_to_expiry, 
                                                                      interest_rate, market_price, steps, 
                                                                      option_type, american)
                        
                        st.subheader("Results")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Implied Volatility", f"{result['implied_volatility']:.4f}")
                        
                        # Store risk assessment for assistant
                        st.session_state.risk_assessment = result['risk_assessment']
                        
                        # Compare with historical volatility
                        try:
                            hist_vol = data_fetcher.calculate_historical_volatility(ticker)
                            st.session_state.hist_vol = hist_vol
                            st.write(f"Historical Volatility: {hist_vol:.4f}")
                            vol_diff = result['implied_volatility'] - hist_vol
                            st.write(f"Implied Volatility is {'higher' if vol_diff > 0 else 'lower'} than historical by {abs(vol_diff):.4f}")
                        except:
                            pass
                
            except Exception as e:
                st.error(f"Calculation error: {str(e)}")
                st.code(traceback.format_exc())

# TAB 2: Market Data
with data_tab:
    st.header("Financial Market Data")
    
    # Stock Data Section
    st.subheader("Stock Data")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        data_ticker = st.text_input("Stock Ticker", "AAPL", key="data_ticker")
        st.session_state.ticker = data_ticker
    with col2:
        data_period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    with col3:
        data_interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
    
    if st.button("Fetch Stock Data"):
        with st.spinner("Fetching data..."):
            try:
                # Fetch stock data
                stock_data = yf.Ticker(data_ticker)
                info = stock_data.info
                
                # Display company info
                if 'longName' in info:
                    company_name = info['longName']
                    st.write(f"## {company_name} ({data_ticker})")
                
                # Create metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    if 'currentPrice' in info:
                        st.metric("Current Price", f"${info['currentPrice']:.2f}")
                    elif 'regularMarketPrice' in info:
                        st.metric("Current Price", f"${info['regularMarketPrice']:.2f}")
                
                with metric_col2:
                    if 'fiftyTwoWeekHigh' in info and 'fiftyTwoWeekLow' in info:
                        st.metric("52 Week Range", f"${info['fiftyTwoWeekLow']:.2f} - ${info['fiftyTwoWeekHigh']:.2f}")
                
                with metric_col3:
                    if 'volume' in info and 'averageVolume' in info:
                        vol_ratio = info['volume'] / info['averageVolume']
                        st.metric("Volume", f"{info['volume']:,}", 
                                delta=f"{vol_ratio:.2f}x avg", delta_color="off")
                
                # Get historical data with technical indicators
                hist = data_fetcher.get_historical_prices(data_ticker, days=180)
                st.session_state.hist_data = hist
                
                if len(hist) > 0:
                    # Create price chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close Price'))
                    
                    # Add moving averages if we have enough data
                    if 'Moving_Avg_20' in hist.columns:
                        fig.add_trace(go.Scatter(x=hist.index, y=hist['Moving_Avg_20'], mode='lines', 
                                                name='20 Day MA', line=dict(color='orange')))
                    
                    if 'Moving_Avg_50' in hist.columns:
                        fig.add_trace(go.Scatter(x=hist.index, y=hist['Moving_Avg_50'], mode='lines', 
                                                name='50 Day MA', line=dict(color='green')))
                    
                    fig.update_layout(title=f"{data_ticker} Price Chart",
                                     xaxis_title="Date",
                                     yaxis_title="Price ($)",
                                     hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Technical indicators
                    st.subheader("Technical Indicators")
                    
                    # RSI
                    if 'RSI' in hist.columns:
                        rsi_fig = go.Figure()
                        rsi_fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], name='RSI'))
                        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                        rsi_fig.update_layout(title="Relative Strength Index (RSI)",
                                             xaxis_title="Date",
                                             yaxis_title="RSI")
                        st.plotly_chart(rsi_fig, use_container_width=True)
                    
                    # Bollinger Bands
                    if 'Bollinger_Upper' in hist.columns and 'Bollinger_Lower' in hist.columns:
                        bb_fig = go.Figure()
                        bb_fig.add_trace(go.Scatter(x=hist.index, y=hist['Bollinger_Upper'], 
                                         name='Upper Band', line=dict(color='red')))
                        bb_fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], 
                                         name='Close Price', line=dict(color='blue')))
                        bb_fig.add_trace(go.Scatter(x=hist.index, y=hist['Bollinger_Lower'], 
                                         name='Lower Band', line=dict(color='green')))
                        bb_fig.update_layout(title="Bollinger Bands",
                                           xaxis_title="Date",
                                           yaxis_title="Price ($)")
                        st.plotly_chart(bb_fig, use_container_width=True)
                    
                    # Volume chart
                    volume_fig = go.Figure()
                    volume_fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volume'))
                    volume_fig.update_layout(title=f"{data_ticker} Trading Volume",
                                           xaxis_title="Date",
                                           yaxis_title="Volume")
                    st.plotly_chart(volume_fig, use_container_width=True)
                    
                    # Calculate and display additional metrics
                    st.subheader("Advanced Metrics")
                    
                    # Calculate volatility
                    daily_returns = hist['Close'].pct_change().dropna()
                    annual_volatility = daily_returns.std() * np.sqrt(252)
                    
                    # Calculate Sharpe ratio (assuming risk-free rate)
                    try:
                        risk_free_rate = data_fetcher.get_risk_free_rate()
                    except:
                        risk_free_rate = config.DEFAULT_RISK_FREE_RATE
                        
                    sharpe_ratio = (daily_returns.mean() * 252 - risk_free_rate) / (daily_returns.std() * np.sqrt(252))
                    
                    # Calculate drawdown
                    rolling_max = hist['Close'].cummax()
                    drawdown = (hist['Close'] - rolling_max) / rolling_max
                    max_drawdown = drawdown.min()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Annualized Volatility", f"{annual_volatility:.2%}")
                    with col2:
                        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                    with col3:
                        st.metric("Max Drawdown", f"{max_drawdown:.2%}")
                    
                    # Provide data as a table
                    st.subheader("Historical Data")
                    st.dataframe(hist)
                    
                    # Download button
                    csv = hist.to_csv().encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{data_ticker}{data_period}{data_interval}.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("No historical data available for the selected parameters")
                
                # Get options data if available
                st.subheader("Options Data")
                try:
                    options_data = data_fetcher.get_option_chain(data_ticker)
                    expiry = options_data['expiry']
                    
                    st.write(f"Options expiring on: {expiry}")
                    
                    # Create tabs for calls and puts
                    calls_tab, puts_tab = st.tabs(["Calls", "Puts"])
                    
                    with calls_tab:
                        calls_df = options_data['calls']
                        st.dataframe(calls_df.style.applymap(
                            lambda x: 'background-color: #ffcccc' if x == 'High' else (
                                'background-color: #ffffcc' if x == 'Medium' else 'background-color: #ccffcc'
                            ), subset=['Risk_Level']))
                    
                    with puts_tab:
                        puts_df = options_data['puts']
                        st.dataframe(puts_df.style.applymap(
                            lambda x: 'background-color: #ffcccc' if x == 'High' else (
                                'background-color: #ffffcc' if x == 'Medium' else 'background-color: #ccffcc'
                            ), subset=['Risk_Level']))
                    
                except Exception as e:
                    st.info(f"Options data not available for {data_ticker}")
                
                # Market sentiment
                st.subheader("Market Sentiment")
                try:
                    sentiment = data_fetcher.get_market_sentiment(data_ticker)
                    st.write(f"Recent news articles: {sentiment['news_count']}")
                    
                    if sentiment['sentiment'] == 'positive':
                        st.success("Overall sentiment: Positive")
                    elif sentiment['sentiment'] == 'negative':
                        st.error("Overall sentiment: Negative")
                    else:
                        st.info("Overall sentiment: Neutral")
                except:
                    st.info("Market sentiment data not available")
                
            except Exception as e:
                st.error(f"Error fetching market data: {str(e)}")
                st.code(traceback.format_exc())

# TAB 3: AI Finance Assistant
with bot_tab:
    st.header("AI Finance Assistant")
    st.markdown("""
    Ask me anything about stocks, options, financial markets, or financial theory.
    I can provide market data, analyze stocks, explain financial concepts, and more.
    """)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Ask something about finance..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = finance_bot.answer_question(prompt)
                st.markdown(response)
            
            # Add AI response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown(f"© {datetime.now().year} Financial Analytics AI Platform | Powered by Streamlit, YFinance")