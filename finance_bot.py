#finance_bot.py
"""
AI Finance Assistant implementation using OpenAI's API.
"""
import openai
import config
from live_data import FinancialDataFetcher

class FinanceAIBot:
    """AI Finance Assistant for answering financial questions."""
    
    def __init__(self):
        """Initialize the finance bot."""
        self.data_fetcher = FinancialDataFetcher()
        try:
            openai.api_key = config.OPENAI_API_KEY
        except:
            pass  # Handle cases where API key is not set
    
    def answer_question(self, question):
        """Answer a financial question using AI and live data."""
        try:
            # First try to handle with specific data queries
            if "price of" in question.lower() or "stock price" in question.lower():
                ticker = self._extract_ticker(question)
                if ticker:
                    price = self.data_fetcher.get_stock_price(ticker)
                    return f"The current price of {ticker} is ${price:.2f}"
            
            if "volatility" in question.lower() and "historical" in question.lower():
                ticker = self._extract_ticker(question)
                if ticker:
                    vol = self.data_fetcher.calculate_historical_volatility(ticker)
                    return f"The historical volatility of {ticker} is {vol:.4f} (annualized)"
            
            # Fall back to general AI response
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a knowledgeable financial analyst. Provide accurate, concise answers to finance questions. When discussing stocks or options, include relevant metrics if possible."},
                    {"role": "user", "content": question}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        
        except Exception as e:
            return f"I encountered an error processing your request: {str(e)}. Please try again later."
    
    def _extract_ticker(self, text):
        """Extract a stock ticker from text."""
        words = text.upper().split()
        for word in words:
            if len(word) <= 5 and word.isalpha():
                return word
        return None
    