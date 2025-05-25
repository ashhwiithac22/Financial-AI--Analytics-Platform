#newton_raphson.py
"""
Enhanced implementation of Newton-Raphson method for solving implied volatility with robust convergence handling.
"""
import numpy as np
from binomial_model import BinomialModel
import config
from live_data import FinancialDataFetcher

class ImpliedVolatilitySolver:
    """
    Class for solving implied volatility using Newton-Raphson method with robust convergence handling.
    """
    
    def __init__(self):
        """Initialize the implied volatility solver."""
        self.binomial_model = BinomialModel()
        self.data_fetcher = FinancialDataFetcher()
    
    def price_option_for_iv(self, S, K, T, r, sigma, steps, option_type, american):
        """Calculate option price for a given volatility."""
        return self.binomial_model.price_option(S, K, T, r, sigma, steps, option_type, american)
    
    def price_derivative(self, S, K, T, r, sigma, steps, option_type, american, h=0.0001):
        """Calculate the derivative of option price with respect to volatility."""
        price_up = self.price_option_for_iv(S, K, T, r, sigma + h, steps, option_type, american)
        price_down = self.price_option_for_iv(S, K, T, r, sigma - h, steps, option_type, american)
        return (price_up - price_down) / (2 * h)
    
    def solve_implied_volatility(self, S, K, T, r, market_price, steps, option_type, american=False):
        """Robust solve for implied volatility using Newton-Raphson method."""
        # Initial guesses with wider range
        sigma_guesses = [0.1, 0.3, 0.5, 1.0, 2.0]
        
        for initial_sigma in sigma_guesses:
            sigma = initial_sigma
            last_diff = float('inf')
            
            for i in range(config.MAX_ITERATIONS):
                try:
                    price = self.price_option_for_iv(S, K, T, r, sigma, steps, option_type, american)
                    price_diff = price - market_price
                    
                    # Check for convergence
                    if abs(price_diff) < config.TOLERANCE:
                        return sigma
                    
                    # Check if we're oscillating or diverging
                    if abs(price_diff) >= abs(last_diff):
                        break  # Try next initial guess
                    
                    last_diff = price_diff
                    
                    vega = self.price_derivative(S, K, T, r, sigma, steps, option_type, american)
                    
                    # Handle near-zero vega
                    if abs(vega) < 1e-8:
                        vega = 1e-8 if vega >= 0 else -1e-8
                    
                    # Calculate new sigma with damping factor
                    damping = 0.5 if abs(price_diff) > 10 else 1.0
                    new_sigma = sigma - damping * price_diff / vega
                    
                    # Ensure sigma stays within reasonable bounds
                    new_sigma = max(0.001, min(new_sigma, 5.0))
                    
                    # Weighted average for stability
                    sigma = 0.7 * sigma + 0.3 * new_sigma
                    
                except Exception as e:
                    break  # Try next initial guess
        
        # If all initial guesses failed, try bisection method as fallback
        return self.bisection_iv(S, K, T, r, market_price, steps, option_type, american)
    
    def bisection_iv(self, S, K, T, r, market_price, steps, option_type, american, max_iter=50, tol=1e-4):
        """Bisection method as fallback for implied volatility calculation."""
        a, b = 0.001, 5.0  # Reasonable volatility bounds
        
        # Check if solution is within bounds
        fa = self.price_option_for_iv(S, K, T, r, a, steps, option_type, american) - market_price
        fb = self.price_option_for_iv(S, K, T, r, b, steps, option_type, american) - market_price
        
        if fa * fb >= 0:
            # No solution in this interval, return best guess
            return (a + b) / 2
        
        for _ in range(max_iter):
            c = (a + b) / 2
            fc = self.price_option_for_iv(S, K, T, r, c, steps, option_type, american) - market_price
            
            if abs(fc) < tol:
                return c
                
            if fc * fa < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
        
        return (a + b) / 2
    
    def calculate_implied_volatility(self, S, K, T, r, market_price, steps, option_type, american=False):
        """
        Calculate implied volatility with robust error handling and fallback methods.
        """
        try:
            # Input validation
            if S <= 0 or K <= 0 or T <= 0 or steps <= 0:
                raise ValueError("Stock price, strike price, time to expiration, and steps must be positive")
            
            if market_price <= 0:
                raise ValueError("Market price must be positive")
            
            # Calculate intrinsic value
            if option_type.lower() == "call":
                intrinsic = max(0, S - K)
            else:
                intrinsic = max(0, K - S)
            
            if market_price < intrinsic - 1e-4:
                raise ValueError(f"Market price ({market_price:.4f}) is below intrinsic value ({intrinsic:.4f})")
            
            # Try Newton-Raphson first
            implied_vol = self.solve_implied_volatility(S, K, T, r, market_price, steps, option_type, american)
            
            # Verify the solution
            calculated_price = self.price_option_for_iv(S, K, T, r, implied_vol, steps, option_type, american)
            if abs(calculated_price - market_price) > 0.1:  # Large error threshold
                # Try bisection if Newton-Raphson result is poor
                implied_vol = self.bisection_iv(S, K, T, r, market_price, steps, option_type, american)
            
            # Final verification
            calculated_price = self.price_option_for_iv(S, K, T, r, implied_vol, steps, option_type, american)
            if abs(calculated_price - market_price) > 0.5:  # Very large error threshold
                raise ValueError("Could not find accurate implied volatility solution")
            
            # Risk assessment
            risk = self.assess_implied_volatility_risk(S, K, T, r, implied_vol, market_price, option_type)
            
            return {
                "implied_volatility": implied_vol,
                "risk_assessment": risk,
                "calculation_error": abs(calculated_price - market_price)
            }
            
        except Exception as e:
            # Provide a reasonable default with warning if all methods fail
            default_vol = 0.3
            risk = {
                "level": "High",
                "reasons": ["Implied volatility calculation failed", "Using default volatility"],
                "recommendation": "Verify inputs and try again",
                "implied_volatility": default_vol
            }
            
            return {
                "implied_volatility": default_vol,
                "risk_assessment": risk,
                "calculation_error": float('nan'),
                "warning": f"Implied volatility calculation failed: {str(e)}. Using default value {default_vol}"
            }
    
    def assess_implied_volatility_risk(self, S, K, T, r, implied_vol, market_price, option_type):
        """Assess the risk based on implied volatility."""
        risk = {
            "level": "Medium",
            "reasons": [],
            "recommendation": "Hold",
            "implied_volatility": implied_vol
        }
        
        # Compare implied vol to historical vol
        try:
            hist_vol = self.data_fetcher.calculate_historical_volatility(S)
            if implied_vol > hist_vol * 1.5:
                risk["reasons"].append(f"Implied vol ({implied_vol:.2f}) significantly higher than historical vol ({hist_vol:.2f})")
                risk["level"] = "High"
                risk["recommendation"] = "Consider selling options"
            elif implied_vol > hist_vol * 1.2:
                risk["reasons"].append(f"Implied vol ({implied_vol:.2f}) higher than historical vol ({hist_vol:.2f})")
                risk["level"] = "Medium-High"
            elif implied_vol < hist_vol * 0.8:
                risk["reasons"].append(f"Implied vol ({implied_vol:.2f}) lower than historical vol ({hist_vol:.2f})")
                risk["level"] = "Low"
                risk["recommendation"] = "Potential buying opportunity"
        except:
            pass
        
        # Absolute volatility levels
        if implied_vol > 0.5:
            risk["reasons"].append("Very high implied volatility (>50%)")
            risk["level"] = "Very High"
            risk["recommendation"] = "Consider selling options"
        elif implied_vol > 0.4:
            risk["reasons"].append("High implied volatility (>40%)")
            risk["level"] = "High"
            risk["recommendation"] = "Consider selling options"
        elif implied_vol < 0.2:
            risk["reasons"].append("Low implied volatility (<20%)")
            risk["level"] = "Low"
            risk["recommendation"] = "Potential buying opportunity"
        
        # Time to expiry consideration
        if T < 0.1:  # Less than 36.5 days
            risk["reasons"].append("Short time to expiry")
            if implied_vol > 0.4:
                risk["level"] = "High"
                risk["recommendation"] = "High risk - consider avoiding"
        
        # Moneyness consideration
        if option_type == "call":
            moneyness = (S - K) / S
        else:
            moneyness = (K - S) / S
            
        if moneyness < -0.2:  # Far out of the money
            risk["reasons"].append("Far out of the money")
            if implied_vol > 0.4:
                risk["level"] = "Very High"
                risk["recommendation"] = "Very risky - avoid buying"
        
        return risk