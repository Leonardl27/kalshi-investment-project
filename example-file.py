#!/usr/bin/env python3
"""
Example script showing how to use the Kalshi Investment Analyzer.
"""

import sys
import os
import json

# Add parent directory to path to import kalshi_analyzer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kalshi_analyzer import KalshiInvestmentAnalyzer

def main():
    """Run a simple investment analysis example."""
    # Load configuration
    try:
        with open("../config.json", "r") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Using default configuration")
        config = {
            "tax_rate": 0.25,
            "kalshi_fee": 0.05,
            "interest_rate": 0.05,
            "risk_aversion": 1.0
        }
    
    # Initialize analyzer
    analyzer = KalshiInvestmentAnalyzer(config)
    
    # Example scenario
    p_your = 0.25       # Your estimated probability
    p_market = 0.39     # Market-implied probability
    days_to_resolve = 45  # Days until market resolution
    investment_amount = 1000  # Amount to invest in dollars
    
    # Analyze "No" bet
    no_analysis = analyzer.analyze_investment(
        p_your=p_your,
        p_market=p_market,
        days_to_resolve=days_to_resolve,
        bet_type="no",
        investment_amount=investment_amount
    )
    
    # Analyze "Yes" bet
    yes_analysis = analyzer.analyze_investment(
        p_your=p_your,
        p_market=p_market,
        days_to_resolve=days_to_resolve,
        bet_type="yes",
        investment_amount=investment_amount
    )
    
    # Compare results
    print("\n===== Investment Analysis Example =====")
    print(f"Your Probability: {p_your:.2f}")
    print(f"Market Probability: {p_market:.2f}")
    print(f"Days to Resolve: {days_to_resolve}")
    print(f"Investment Amount: ${investment_amount:.2f}")
    
    print("\n----- 'No' Bet Analysis -----")
    for key, value in no_analysis.items():
        if isinstance(value, float):
            if key in ["Your Probability", "Market Probability", "Probability Edge", "Confidence"]:
                print(f"{key}: {value:.2f}")
            elif "Return" in key or "EV" in key:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print("\n----- 'Yes' Bet Analysis -----")
    for key, value in yes_analysis.items():
        if isinstance(value, float):
            if key in ["Your Probability", "Market Probability", "Probability Edge", "Confidence"]:
                print(f"{key}: {value:.2f}")
            elif "Return" in key or "EV" in key:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Recommendation
    if no_analysis["Decision"] == "Invest" and yes_analysis["Decision"] == "Don't Invest":
        print("\nRecommendation: Bet 'No'")
    elif no_analysis["Decision"] == "Don't Invest" and yes_analysis["Decision"] == "Invest":
        print("\nRecommendation: Bet 'Yes'")
    elif no_analysis["Decision"] == "Invest" and yes_analysis["Decision"] == "Invest":
        if no_analysis["Expected Profit"] > yes_analysis["Expected Profit"]:
            print("\nRecommendation: Bet 'No' (higher expected profit)")
        else:
            print("\nRecommendation: Bet 'Yes' (higher expected profit)")
    else:
        print("\nRecommendation: Don't Invest (neither bet has positive expected value)")


if __name__ == "__main__":
    main()
