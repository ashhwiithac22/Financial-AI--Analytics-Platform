# Financial Analytics AI Platform

A Python-based web application for financial analytics, combining option pricing models, implied volatility calculations, live data integration, and AI-powered financial insights.

## Features

### 1. Binomial Option Pricing Model
- Implementation of the Cox-Ross-Rubinstein binomial pricing model
- Support for both European and American options
- Calculation of option Greeks (Delta, Gamma, Theta, Vega, Rho)

### 2. Implied Volatility Solver
- Uses Newton-Raphson numerical method to solve for implied volatility
- Leverages JAX for automatic differentiation
- Fast and accurate convergence

### 3. Live Financial Data Integration
- Real-time stock price data
- Historical price data for volatility calculations
- Options chain data
- Risk-free rate estimation

### 4. AI-Powered Financial Assistant
- Integrated with OpenAI's GPT-4
- Provides market analysis and financial insights
- Answers finance-related questions using real-time data

### 5. Interactive Streamlit UI
- Intuitive interface for option pricing and implied volatility calculations
- Live market data visualization with interactive charts
- Conversational interface for the financial AI assistant

## Project Structure

```
finance_ai_project/
├── app.py                 # Streamlit UI main file
├── binomial_model.py      # Binomial Option Pricing
├── newton_raphson.py      # Implied Volatility Solver (with AutoDiff)
├── live_data.py           # Live stock/option data fetcher
├── finance_bot.py         # Gen AI Chatbot integration using OpenAI API
├── config.py              # API keys and constants
├── requirements.txt       # All dependencies
└── README.md              # Project description
```

## Setup and Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
   ```
4. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

### Option Pricing & Implied Volatility Calculator
- Enter stock ticker to automatically fetch current price
- Input option parameters (strike, expiry, etc.)
- Choose calculation mode (price option or find implied volatility)
- View results with interactive visualization

### Market Data
- Search for any stock ticker
- View current price, historical price chart, and key statistics
- Access available options data

### Financial AI Assistant
- Ask financial questions in natural language
- Get responses based on financial knowledge and live market data
- Clear conversation history as needed

## Requirements

- Python 3.8 or higher
- Libraries: numpy, pandas, yfinance, jax, streamlit, openai, requests, python-dotenv, plotly

## License

This project is licensed under the MIT License - see the LICENSE file for details.