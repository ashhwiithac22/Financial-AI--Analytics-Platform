#live_data.py
"""
Enhanced module for fetching live financial data with additional calculations.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import config
from scipy.stats import norm

class FinancialDataFetcher:
    """Enhanced class to fetch and analyze live financial data."""
 
    def __init__(self):
        """Initialize the data fetcher with caching."""
        self.cache = {}
        self.cache_timestamp = {}
        self.cache_timeout = 300  # 5 minutes
        self.failed_tickers = set()
        self.failure_timestamp = {}
        self.failure_timeout = 3600  # 1 hour

    def _is_cache_valid(self, key):
        """Check if cached data is still valid."""
        if key not in self.cache_timestamp:
            return False
        elapsed = (datetime.now() - self.cache_timestamp[key]).total_seconds()
        return elapsed < self.cache_timeout

    def _mark_ticker_failed(self, ticker):
        """Mark a ticker as failed to avoid repeated failed requests."""
        self.failed_tickers.add(ticker)
        self.failure_timestamp[ticker] = datetime.now()

    def _is_ticker_valid(self, ticker):
        """Check if a ticker is valid and fetchable."""
        ticker = ticker.upper()
        if ticker in self.failed_tickers:
            elapsed = (datetime.now() - self.failure_timestamp[ticker]).total_seconds()
            if elapsed < self.failure_timeout:
                return False
            self.failed_tickers.remove(ticker)

        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if 'regularMarketPrice' in info or 'previousClose' in info:
                return True
        except:
            pass

        self._mark_ticker_failed(ticker)
        return False

    def get_stock_price(self, ticker):
        """Get current stock price with caching."""
        ticker = ticker.upper()
        cache_key = f"price_{ticker}"

        if cache_key in self.cache and self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        if not self._is_ticker_valid(ticker):
            raise ValueError(f"Invalid ticker symbol: {ticker}")

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                price = info['regularMarketPrice']
            elif 'previousClose' in info and info['previousClose'] is not None:
                price = info['previousClose']
            else:
                hist = stock.history(period="1d")
                price = hist['Close'].iloc[-1] if not hist.empty else 100.0

            self.cache[cache_key] = price
            self.cache_timestamp[cache_key] = datetime.now()
            return price
        except Exception as e:
            self._mark_ticker_failed(ticker)
            raise ValueError(f"Failed to get price for {ticker}: {str(e)}")

    def get_risk_free_rate(self):
        """Get current risk-free rate (10-year Treasury yield)."""
        cache_key = "risk_free_rate"
        if cache_key in self.cache and self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        try:
            for symbol in ["^IRX", "^TNX"]:
                treasury = yf.Ticker(symbol)
                data = treasury.history(period="1d")
                if not data.empty:
                    rate = data['Close'].iloc[-1] / 100.0
                    self.cache[cache_key] = rate
                    self.cache_timestamp[cache_key] = datetime.now()
                    return rate
        except:
            pass

        return config.DEFAULT_RISK_FREE_RATE

    def get_historical_prices(self, ticker, days=252):
        """Get historical prices with enhanced technical indicators."""
        ticker = ticker.upper()
        cache_key = f"hist_{ticker}_{days}"

        if cache_key in self.cache and self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        if not self._is_ticker_valid(ticker):
            raise ValueError(f"Invalid ticker symbol: {ticker}")

        try:
            end = datetime.now()
            start = end - timedelta(days=days * 2)
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))

            if hist.empty:
                raise ValueError(f"No historical data for {ticker}")

            # Calculate enhanced technical indicators
            hist['Daily_Return'] = hist['Close'].pct_change()
            hist['Cumulative_Return'] = (1 + hist['Daily_Return']).cumprod() - 1
            
            # Moving Averages
            hist['Moving_Avg_20'] = hist['Close'].rolling(20).mean()
            hist['Moving_Avg_50'] = hist['Close'].rolling(50).mean()
            hist['Moving_Avg_200'] = hist['Close'].rolling(200).mean()
            
            # Volatility measures
            hist['Daily_Volatility'] = hist['Daily_Return'].rolling(20).std()
            hist['Annualized_Volatility'] = hist['Daily_Volatility'] * np.sqrt(252)
            
            # Momentum indicators
            hist['RSI'] = self._calculate_rsi(hist['Close'])
            hist['MACD'], hist['MACD_Signal'] = self._calculate_macd(hist['Close'])
            
            # Bollinger Bands
            hist['Bollinger_Upper'], hist['Bollinger_Lower'] = self._calculate_bollinger_bands(hist['Close'])
            
            # Volume analysis
            hist['Volume_MA_20'] = hist['Volume'].rolling(20).mean()
            hist['Volume_Ratio'] = hist['Volume'] / hist['Volume_MA_20']
            
            # Support/Resistance levels
            hist['Support'] = hist['Close'].rolling(20).min()
            hist['Resistance'] = hist['Close'].rolling(20).max()
            
            # Remove early periods with NaN values
            hist = hist.iloc[200:].copy()

            self.cache[cache_key] = hist
            self.cache_timestamp[cache_key] = datetime.now()
            return hist
        except Exception as e:
            self._mark_ticker_failed(ticker)
            raise ValueError(f"Error getting historical data for {ticker}: {str(e)}")

    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index (RSI)."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, macd_signal

    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands."""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        return upper_band, lower_band

    def calculate_historical_volatility(self, ticker, days=30):
        """Calculate annualized historical volatility with enhanced method."""
        ticker = ticker.upper()
        cache_key = f"vol_{ticker}_{days}"

        if cache_key in self.cache and self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        try:
            data = self.get_historical_prices(ticker, max(days, 50))
            returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
            
            if len(returns) < 5:
                return 0.3
                
            # Calculate multiple volatility measures
            simple_vol = returns.std() * np.sqrt(252)
            ewma_vol = returns.ewm(span=days).std().iloc[-1] * np.sqrt(252)
            parkinson_vol = self._parkinson_volatility(data['High'], data['Low'])
            
            # Use weighted average of different methods
            annual_vol = (simple_vol * 0.5 + ewma_vol * 0.3 + parkinson_vol * 0.2)

            self.cache[cache_key] = annual_vol
            self.cache_timestamp[cache_key] = datetime.now()
            return annual_vol
        except:
            return 0.3

    def _parkinson_volatility(self, high, low):
        """Calculate Parkinson volatility estimator."""
        log_hl = np.log(high / low)
        return np.sqrt(252 / (4 * len(log_hl) * np.sum(log_hl ** 2)))

    def get_option_chain(self, ticker):
        """Get option chain data with enhanced risk assessment."""
        ticker = ticker.upper()
        cache_key = f"options_{ticker}"

        if cache_key in self.cache and self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        if not self._is_ticker_valid(ticker):
            raise ValueError(f"Invalid ticker symbol: {ticker}")

        try:
            stock = yf.Ticker(ticker)
            options = stock.options
            if not options:
                raise ValueError(f"No options available for {ticker}")

            expiry = options[0]
            chain = stock.option_chain(expiry)

            # Add enhanced risk assessment to calls and puts
            calls = chain.calls
            puts = chain.puts
            
            # Calculate probability of being in the money
            current_price = self.get_stock_price(ticker)
            time_to_expiry = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.now()).days / 365.0
            risk_free_rate = self.get_risk_free_rate()
            
            # Add probability calculations
            calls['Probability_ITM'] = calls.apply(
                lambda row: self._probability_itm(current_price, row['strike'], time_to_expiry, risk_free_rate, row['impliedVolatility'], 'call'), 
                axis=1
            )
            puts['Probability_ITM'] = puts.apply(
                lambda row: self._probability_itm(current_price, row['strike'], time_to_expiry, risk_free_rate, row['impliedVolatility'], 'put'), 
                axis=1
            )
            
            # Enhanced risk assessment
            calls['Risk_Level'] = self._assess_option_risk(calls, 'call', current_price)
            puts['Risk_Level'] = self._assess_option_risk(puts, 'put', current_price)

            result = {
                'expiry': expiry,
                'calls': calls,
                'puts': puts,
                'current_price': current_price,
                'time_to_expiry': time_to_expiry
            }

            self.cache[cache_key] = result
            self.cache_timestamp[cache_key] = datetime.now()
            return result
        except Exception as e:
            raise ValueError(f"Failed to fetch options for {ticker}: {str(e)}")

    def _probability_itm(self, S, K, T, r, sigma, option_type):
        """Calculate probability of option being in the money at expiration."""
        if T <= 0 or sigma <= 0:
            return 0.0
            
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        if option_type == 'call':
            return norm.cdf(d2)
        else:  # put
            return norm.cdf(-d2)

    def _assess_option_risk(self, options_df, option_type, current_price):
        """Enhanced risk assessment for options."""
        risk_levels = []
        
        for _, row in options_df.iterrows():
            strike = row['strike']
            last_price = row['lastPrice']
            implied_vol = row['impliedVolatility']
            prob_itm = row.get('Probability_ITM', 0.5)
            
            # Calculate moneyness
            if option_type == 'call':
                moneyness = (current_price - strike) / current_price
            else:  # put
                moneyness = (strike - current_price) / current_price
            
            # Enhanced risk assessment logic
            risk_score = 0
            
            # Volatility component (0-40 points)
            if implied_vol > 0.6:
                risk_score += 40
            elif implied_vol > 0.5:
                risk_score += 30
            elif implied_vol > 0.4:
                risk_score += 20
            elif implied_vol > 0.3:
                risk_score += 10
                
            # Moneyness component (0-30 points)
            if moneyness < -0.2:  # Far OTM
                risk_score += 30
            elif moneyness < -0.1:  # OTM
                risk_score += 20
            elif moneyness < 0:  # Slightly OTM
                risk_score += 10
            elif moneyness > 0.2:  # Deep ITM
                risk_score += 0
            elif moneyness > 0.1:  # ITM
                risk_score += 5
                
            # Probability ITM component (0-20 points)
            if prob_itm < 0.2:
                risk_score += 20
            elif prob_itm < 0.3:
                risk_score += 15
            elif prob_itm < 0.4:
                risk_score += 10
            elif prob_itm < 0.5:
                risk_score += 5
                
            # Price component (0-10 points)
            if last_price < 0.5:
                risk_score += 10
            elif last_price < 1.0:
                risk_score += 5
                
            # Determine risk level based on total score
            if risk_score >= 70:
                risk = 'Very High'
            elif risk_score >= 55:
                risk = 'High'
            elif risk_score >= 40:
                risk = 'Medium-High'
            elif risk_score >= 25:
                risk = 'Medium'
            elif risk_score >= 10:
                risk = 'Medium-Low'
            else:
                risk = 'Low'
            
            risk_levels.append(risk)
        
        return risk_levels

    def get_market_sentiment(self, ticker):
        """Get enhanced market sentiment for a given ticker."""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            return {
                'news_count': len(news),
                'sentiment': self._analyze_news_sentiment(news),
                'news_samples': [n['title'] for n in news[:3]] if news else []
            }
        except:
            return {'news_count': 0, 'sentiment': 'neutral', 'news_samples': []}

    def _analyze_news_sentiment(self, news_items):
        """Enhanced news sentiment analysis."""
        if not news_items:
            return 'neutral'
        
        positive_words = ['buy', 'strong', 'growth', 'positive', 'bullish', 'beat', 'raise', 'upgrade']
        negative_words = ['sell', 'weak', 'decline', 'negative', 'bearish', 'miss', 'cut', 'downgrade']
        
        positive_count = 0
        negative_count = 0
        
        for item in news_items:
            title = item.get('title', '').lower()
            summary = item.get('summary', '').lower()
            
            text = f"{title} {summary}"
            
            for word in positive_words:
                if word in text:
                    positive_count += 1
            for word in negative_words:
                if word in text:
                    negative_count += 1
        
        if positive_count == 0 and negative_count == 0:
            return 'neutral'
            
        sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
        
        if sentiment_score > 0.3:
            return 'positive'
        elif sentiment_score < -0.3:
            return 'negative'
        return 'neutral'

    def calculate_support_resistance(self, ticker, lookback=20):
        """Calculate support and resistance levels."""
        try:
            hist = self.get_historical_prices(ticker, lookback * 2)
            recent = hist.tail(lookback)
            
            support = recent['Low'].min()
            resistance = recent['High'].max()
            
            return {
                'support': support,
                'resistance': resistance,
                'current_price': hist['Close'].iloc[-1],
                'distance_to_support': (hist['Close'].iloc[-1] - support) / hist['Close'].iloc[-1],
                'distance_to_resistance': (resistance - hist['Close'].iloc[-1]) / hist['Close'].iloc[-1]
            }
        except:
            return None

    def calculate_expected_move(self, ticker, days=30):
        """Calculate expected price move based on options market."""
        try:
            chain = self.get_option_chain(ticker)
            current_price = chain['current_price']
            time_to_expiry = chain['time_to_expiry']
            
            # Get at-the-money options
            atm_call = chain['calls'].iloc[(chain['calls']['strike'] - current_price).abs().argsort()[:1]]
            atm_put = chain['puts'].iloc[(chain['puts']['strike'] - current_price).abs().argsort()[:1]]
            
            if len(atm_call) == 0 or len(atm_put) == 0:
                return None
                
            iv = (atm_call['impliedVolatility'].values[0] + atm_put['impliedVolatility'].values[0]) / 2
            
            expected_move = current_price * iv * np.sqrt(time_to_expiry)
            
            return {
                'expected_move': expected_move,
                'upper_bound': current_price + expected_move,
                'lower_bound': current_price - expected_move,
                'implied_volatility': iv,
                'probability_68': 0.68,  # 1 standard deviation
                'probability_95': 0.95   # 2 standard deviations
            }
        except:
            return None

    def clear_cache(self):
        """Clear all cached data."""
        self.cache = {}
        self.cache_timestamp = {}
        self.failed_tickers = set()
        self.failure_timestamp = {}
        