import json
from investment import kalshi_investment_decision_no_bet
from kalshi_api import fetch_kalshi_market

# Load API Key and Configuration
api_key = input("Enter your Kalshi API Key: ")
private_key_path = input("Enter the path to your private key file: ")
market_path = input("Enter the Kalshi API market path (e.g., '/trade-api/v2/portfolio/balance'): ")

# Fetch market data
market_data = fetch_kalshi_market(api_key, private_key_path, market_path)
print("Market Data:", market_data)

# User Inputs
p_your = float(input("Enter your estimated probability (as a decimal, e.g., 0.05 for 5%): "))
p_market = float(input("Enter the market-implied probability (as a decimal, e.g., 0.39 for 39%): "))
days_to_resolve = int(input("Enter the number of days until market resolution: "))

# Run investment decision function
investment_decision_no_bet = kalshi_investment_decision_no_bet(p_your, p_market, days_to_resolve)

# Display the result
print("\nInvestment Decision:")
for key, value in investment_decision_no_bet.items():
    print(f"{key}: {value}")
