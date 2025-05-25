#config.py
# Application configuration
APP_TITLE = "Financial Analytics AI Platform"
APP_DESCRIPTION = """
A comprehensive platform for options pricing, implied volatility calculation, 
and financial market analysis powered by AI.
"""

# Default parameters
DEFAULT_RISK_FREE_RATE = 0.03
INITIAL_VOLATILITY_GUESS = 0.3
MAX_ITERATIONS = 100
TOLERANCE = 1e-6

# Risk assessment thresholds
HIGH_VOLATILITY_THRESHOLD = 0.4
LOW_VOLATILITY_THRESHOLD = 0.2
SHORT_EXPIRY_THRESHOLD = 0.1  # years
FAR_OTM_THRESHOLD = -0.2
DEEP_ITM_THRESHOLD = 0.2

# Visualization settings
MAX_POINTS_FOR_GRAPH = 100
COLORS = {
    'high_risk': '#FF6B6B',
    'medium_risk': '#FFD166',
    'low_risk': '#06D6A0',
    'very_high_risk': '#EF476F',
    'very_low_risk': '#118AB2'
}
