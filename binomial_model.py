#binomial_model.py
import streamlit as st
import numpy as np
import math
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go

# Original Binomial Model class preserved intact
class BinomialModel:
    """
    A class implementing the binomial option pricing model with risk assessment.
    """
    
    def init(self):
        """Initialize the binomial model."""
        pass
    
    def _validate_inputs(self, S, K, T, r, sigma, steps, option_type):
        """Validate inputs for the binomial model."""
        if S <= 0:
            raise ValueError("Stock price must be positive")
        if K <= 0:
            raise ValueError("Strike price must be positive")
        if T <= 0:
            raise ValueError("Time to expiration must be positive")
        if sigma <= 0:
            raise ValueError("Volatility must be positive")
        if steps <= 0 or not isinstance(steps, int):
            raise ValueError("Steps must be a positive integer")
        if option_type.lower() not in ["call", "put"]:
            raise ValueError("Option type must be 'call' or 'put'")
    
    def price_option(self, S, K, T, r, sigma, steps, option_type, american=False):
        """
        Calculate option price using the binomial model with risk assessment.
        """
        self._validate_inputs(S, K, T, r, sigma, steps, option_type)
        option_type = option_type.lower()
        
        dt = T / steps
        u = math.exp(sigma * math.sqrt(dt))
        d = 1 / u
        p = (math.exp(r * dt) - d) / (u - d)
        discount = math.exp(-r * dt)
        
        stock_prices = [S * (u ** (steps - i)) * (d ** i) for i in range(steps + 1)]
        
        if option_type == "call":
            option_values = [max(0, price - K) for price in stock_prices]
        else:  # put
            option_values = [max(0, K - price) for price in stock_prices]
        
        for step in range(steps - 1, -1, -1):
            for i in range(step + 1):
                current_price = S * (u ** (step - i)) * (d ** i)
                option_value = discount * (p * option_values[i] + (1 - p) * option_values[i + 1])
                
                if american:
                    if option_type == "call":
                        exercise_value = max(0, current_price - K)
                    else:  # put
                        exercise_value = max(0, K - current_price)
                    option_value = max(option_value, exercise_value)
                
                option_values[i] = option_value
        
        return option_values[0]
    
    def calculate_option_greeks(self, S, K, T, r, sigma, steps, option_type, american=False):
        """Calculate option Greeks with risk assessment."""
        self._validate_inputs(S, K, T, r, sigma, steps, option_type)
        
        dS = S * 0.01
        dsigma = 0.01
        dT = 1 / 365
        
        price = self.price_option(S, K, T, r, sigma, steps, option_type, american)
        
        price_up = self.price_option(S + dS, K, T, r, sigma, steps, option_type, american)
        price_down = self.price_option(S - dS, K, T, r, sigma, steps, option_type, american)
        delta = (price_up - price_down) / (2 * dS)
        
        gamma = (price_up - 2 * price + price_down) / (dS ** 2)
        
        if T <= dT:
            theta = 0
        else:
            price_tm = self.price_option(S, K, T - dT, r, sigma, steps, option_type, american)
            theta = (price_tm - price) / dT
        
        price_vol_up = self.price_option(S, K, T, r, sigma + dsigma, steps, option_type, american)
        price_vol_down = self.price_option(S, K, T, r, sigma - dsigma, steps, option_type, american)
        vega = (price_vol_up - price_vol_down) / (2 * dsigma)
        
        dr = 0.001
        price_r_up = self.price_option(S, K, T, r + dr, sigma, steps, option_type, american)
        price_r_down = self.price_option(S, K, T, r - dr, sigma, steps, option_type, american)
        rho = (price_r_up - price_r_down) / (2 * dr)
        
        # Risk assessment
        risk = self.assess_option_risk(S, K, T, r, sigma, price, option_type, delta, gamma, theta, vega)
        
        return {
            "price": price,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho,
            "risk_assessment": risk
        }
    
    def assess_option_risk(self, S, K, T, r, sigma, price, option_type, delta, gamma, theta, vega):
        """Assess the risk level of an option position."""
        intrinsic = S - K if option_type == "call" else K - S
        time_value = price - max(0, intrinsic)
        
        risk = {
            "level": "Medium",
            "reasons": [],
            "recommendation": "Hold",
            "time_to_expiry": T
        }
        
        # High volatility risk
        if sigma > 0.5:
            risk["reasons"].append("High volatility (>50%)")
            risk["level"] = "High"
        elif sigma > 0.3:
            risk["reasons"].append("Elevated volatility (>30%)")
        
        # Time decay risk
        if T < 0.1:  # Less than 36.5 days
            risk["reasons"].append("Short time to expiry")
            if theta < -0.05:
                risk["reasons"].append("High time decay")
                risk["level"] = "High"
        
        # Moneyness risk
        if option_type == "call":
            moneyness = (S - K) / S
        else:
            moneyness = (K - S) / S
            
        if moneyness < -0.1:  # Far out of the money
            risk["reasons"].append("Far out of the money")
            risk["level"] = "High"
        elif moneyness < 0:  # Out of the money
            risk["reasons"].append("Out of the money")
            if risk["level"] != "High":
                risk["level"] = "Medium-High"
        
        # Gamma risk
        if gamma > 0.1:
            risk["reasons"].append("High gamma (price sensitivity changes rapidly)")
        
        # Generate recommendation
        if risk["level"] == "High":
            if option_type == "call":
                risk["recommendation"] = "Consider selling" if price > intrinsic else "Avoid"
            else:
                risk["recommendation"] = "Consider selling" if price > intrinsic else "Avoid"
        elif risk["level"] in ["Medium-High", "Medium"]:
            if time_value > price * 0.3:
                risk["recommendation"] = "Consider selling" if theta < -0.03 else "Hold"
            else:
                risk["recommendation"] = "Potential buying opportunity" if moneyness > 0.1 else "Hold"
        else:
            if moneyness > 0.1 and time_value < price * 0.2:
                risk["recommendation"] = "Good buying opportunity"
            else:
                risk["recommendation"] = "Hold"
        
        return risk
    
    def days_to_years(self, days):
        """Convert days to years for option pricing."""
        return days / 365.0
    
    def date_to_years(self, expiry_date):
        expiry = datetime.strptime(expiry_date, "%Y-%m-%d")
        now = datetime.now()
        days = (expiry - now).days + (expiry - now).seconds / (24 * 3600)
        return max(0, self.days_to_years(days))


# AI Assistant Chat Responses
class RiskAssistant:
    def init(self):
        self.risk_data = None
        
    def update_risk_data(self, risk_data):
        self.risk_data = risk_data
        
    def get_risk_level_response(self):
        if not self.risk_data:
            return "I don't have any risk data yet. Please calculate an option price first."
            
        level = self.risk_data["level"]
        recommendation = self.risk_data["recommendation"]
        
        responses = {
            "High": f"Your option position has a HIGH risk level. {recommendation}.",
            "Medium-High": f"Your option position has a MEDIUM-HIGH risk level. {recommendation}.",
            "Medium": f"Your option position has a MEDIUM risk level. {recommendation}.",
            "Low": f"Your option position has a LOW risk level. {recommendation}."
        }
        
        return responses.get(level, "Risk level unknown.")
        
    def get_risk_reasons(self):
        if not self.risk_data:
            return "I don't have any risk data yet. Please calculate an option price first."
            
        reasons = self.risk_data["reasons"]
        
        if not reasons:
            return "There are no specific risk factors identified for this position."
            
        response = "The risk assessment is based on these factors:\n\n"
        for i, reason in enumerate(reasons, 1):
            response += f"{i}. {reason}\n"
            
        return response
        
    def get_recommendation(self):
        if not self.risk_data:
            return "I don't have any risk data yet. Please calculate an option price first."
            
        recommendation = self.risk_data["recommendation"]
        time_to_expiry = self.risk_data["time_to_expiry"]
        days = int(time_to_expiry * 365)
        
        return f"Based on the risk assessment, my recommendation is: {recommendation}. You have approximately {days} days until expiration."
        
    def what_is_binomial_model(self):
        return """The Binomial Option Pricing Model is a numerical method for calculating option prices by simulating possible price paths of the underlying asset.

Key features:
• Uses a "binomial tree" where the stock price can move up or down at each step
• Accounts for volatility, interest rates, and time to expiration
• Can price both American and European options
• More accurate with more steps in the calculation

It's particularly useful for pricing American options that can be exercised before expiration."""

    def explain_greeks(self):
        return """Option Greeks measure how option prices respond to changes in market conditions:

𝐃𝐞𝐥𝐭𝐚: How much the option price changes when the underlying stock price changes by $1.

𝐆𝐚𝐦𝐦𝐚: How much the delta changes when the stock price changes by $1.

𝐓𝐡𝐞𝐭𝐚: How much the option price decreases each day as expiration approaches.

𝐕𝐞𝐠𝐚: How much the option price changes when volatility increases by 1%.

𝐑𝐡𝐨: How much the option price changes when interest rates increase by 1%."""


# Set up the Streamlit app
def main():
    st.set_page_config(layout="wide", page_title="Option Pricing with AI Assistant")
    
    # Initialize the model and assistant
    if 'model' not in st.session_state:
        st.session_state.model = BinomialModel()
    if 'assistant' not in st.session_state:
        st.session_state.assistant = RiskAssistant()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'last_calculation' not in st.session_state:
        st.session_state.last_calculation = None
    
    # App title and description
    st.title("Options Risk Calculator")
    
    # Create two columns for layout - calculator on left, chat on right
    col1, col2 = st.columns([6, 4])
    
    with col1:
        st.header("Option Pricing Calculator")
        st.write("Enter the parameters to calculate option price and risk assessment")
        
        # Input parameters
        with st.form("option_form"):
            stock_price = st.number_input("Current Stock Price ($)", min_value=0.01, value=100.0, step=1.0)
            strike_price = st.number_input("Strike Price ($)", min_value=0.01, value=100.0, step=1.0)
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                time_input_type = st.radio("Time Input Type", ["Days to Expiry", "Expiry Date"])
            
            with col_t2:
                if time_input_type == "Days to Expiry":
                    days = st.number_input("Days to Expiry", min_value=1, value=30, step=1)
                    time_to_expiry = days / 365.0
                else:
                    expiry_date = st.date_input("Expiry Date", value=datetime.now() + timedelta(days=30))
                    time_to_expiry = (expiry_date - datetime.now().date()).days / 365.0
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                risk_free_rate = st.number_input("Risk-Free Rate (%)", min_value=0.0, value=3.0, step=0.1) / 100.0
            with col_b:
                volatility = st.number_input("Volatility (%)", min_value=1.0, value=30.0, step=1.0) / 100.0
            with col_c:
                steps = st.number_input("Number of Steps", min_value=5, value=50, step=5)
            
            col_o1, col_o2 = st.columns(2)
            with col_o1:
                option_type = st.radio("Option Type", ["Call", "Put"])
            with col_o2:
                exercise_style = st.radio("Exercise Style", ["European", "American"])
            
            submitted = st.form_submit_button("Calculate")
            
            if submitted:
                try:
                    american = exercise_style == "American"
                    
                    # Calculate price and Greeks
                    result = st.session_state.model.calculate_option_greeks(
                        stock_price, strike_price, time_to_expiry, risk_free_rate, 
                        volatility, steps, option_type, american
                    )
                    
                    # Store the calculation result
                    st.session_state.last_calculation = {
                        "stock_price": stock_price,
                        "strike_price": strike_price,
                        "time_to_expiry": time_to_expiry,
                        "risk_free_rate": risk_free_rate,
                        "volatility": volatility,
                        "option_type": option_type,
                        "exercise_style": exercise_style,
                        "result": result
                    }
                    
                    # Update assistant with new risk data
                    st.session_state.assistant.update_risk_data(result["risk_assessment"])
                    
                except Exception as e:
                    st.error(f"Error in calculation: {str(e)}")
        
        # Display results if available
        if st.session_state.last_calculation:
            result = st.session_state.last_calculation["result"]
            risk = result["risk_assessment"]
            
            st.subheader("Pricing Results")
            
            # Create metrics display
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Option Price", f"${result['price']:.4f}")
            with col_m2:
                risk_color = {"High": "red", "Medium-High": "orange", "Medium": "yellow", "Low": "green"}
                risk_level = risk["level"]
                st.metric("Risk Level", risk_level)
            with col_m3:
                days_to_expiry = int(risk["time_to_expiry"] * 365)
                st.metric("Days to Expiry", days_to_expiry)
            with col_m4:
                st.metric("Recommendation", risk["recommendation"])
            
            # Greeks table
            st.subheader("Greeks")
            greeks_df = pd.DataFrame({
                "Greek": ["Delta", "Gamma", "Theta", "Vega", "Rho"],
                "Value": [
                    f"{result['delta']:.4f}",
                    f"{result['gamma']:.4f}",
                    f"{result['theta']:.4f}",
                    f"{result['vega']:.4f}",
                    f"{result['rho']:.4f}"
                ]
            })
            st.table(greeks_df)
            
            # Risk factors
            st.subheader("Risk Factors")
            if risk["reasons"]:
                for reason in risk["reasons"]:
                    st.write(f"• {reason}")
            else:
                st.write("No specific risk factors identified.")
                
            # Add a simple payoff diagram
            st.subheader("Option Payoff at Expiration")
            
            S = st.session_state.last_calculation["stock_price"]
            K = st.session_state.last_calculation["strike_price"]
            option_type = st.session_state.last_calculation["option_type"]
            
            # Generate price range for x-axis (stock prices)
            price_range = np.linspace(K * 0.7, K * 1.3, 100)
            
            # Calculate payoff
            if option_type == "Call":
                payoffs = np.maximum(price_range - K, 0)
            else:  # Put
                payoffs = np.maximum(K - price_range, 0)
            
            # Create plotly figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=price_range, y=payoffs, mode='lines', name='Payoff'))
            fig.add_trace(go.Scatter(x=[S], y=[0], mode='markers', name='Current Price', 
                                    marker=dict(size=10, color='red')))
            
            # Add strike price vertical line
            fig.add_shape(type="line", x0=K, y0=0, x1=K, y1=max(payoffs),
                        line=dict(color="gray", width=1, dash="dash"))
            
            fig.update_layout(
                title=f"{option_type} Option Payoff",
                xaxis_title="Stock Price at Expiration",
                yaxis_title="Profit/Loss",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # AI Assistant section
    with col2:
        st.header("Risk AI Assistant")
        
        # Assistant chat interface
        st.subheader("Chat with Risk Assistant")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.write(f"You: {message['content']}")
                else:
                    st.write(f"Risk Assistant: {message['content']}")
        
        # Question buttons
        st.write("Ask a question:")
        col_q1, col_q2 = st.columns(2)
        
        with col_q1:
            if st.button("What's my risk level?"):
                response = st.session_state.assistant.get_risk_level_response()
                st.session_state.chat_history.append({"role": "user", "content": "What's my risk level?"})
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.experimental_rerun()
                
            if st.button("Why this risk level?"):
                response = st.session_state.assistant.get_risk_reasons()
                st.session_state.chat_history.append({"role": "user", "content": "Why is this the risk level?"})
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.experimental_rerun()
        
        with col_q2:
            if st.button("What's your recommendation?"):
                response = st.session_state.assistant.get_recommendation()
                st.session_state.chat_history.append({"role": "user", "content": "What's your recommendation?"})
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.experimental_rerun()
                
            if st.button("Explain option Greeks"):
                response = st.session_state.assistant.explain_greeks()
                st.session_state.chat_history.append({"role": "user", "content": "Explain option Greeks"})
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.experimental_rerun()
        
        # Additional questions
        col_q3, col_q4 = st.columns(2)
        with col_q3:
            if st.button("What is the Binomial Model?"):
                response = st.session_state.assistant.what_is_binomial_model()
                st.session_state.chat_history.append({"role": "user", "content": "What is the Binomial Model?"})
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.experimental_rerun()
        
        with col_q4:
            if st.button("Clear chat history"):
                st.session_state.chat_history = []
                st.experimental_rerun()
        
        # Add some information about the assistant
        with st.expander("About Risk Assistant"):
            st.write("""
            This Risk Assistant helps you understand the options risk assessment provided by the calculator.
            
            The assistant can:
            - Explain your current risk level
            - Provide detailed reasons for the risk assessment
            - Give recommendations based on the risk profile
            - Explain key options concepts and metrics
            
            All assessments are based on the Binomial Option Pricing Model and standard risk metrics.
            """)

if __name__ == "__main__":
    main()