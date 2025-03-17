import json

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

TAX_RATE = config["tax_rate"]
KALSHI_FEE = config["kalshi_fee"]
INTEREST_RATE = config["interest_rate"]

def kalshi_investment_decision_no_bet(p_your, p_market, days_to_resolve):
    """
    Determines whether to invest in a Kalshi market by betting on "No" (shorting Yes),
    taking into account expected value, taxes, fees, and opportunity cost.
    """
    payout = (1 / (1 - p_market)) - 1  # Profit per $1 bet for No
    ev = ((1 - p_your) * payout) - (p_your * 1)  # Adjusting for betting against the event
    
    ev_after_tax = ev * (1 - TAX_RATE)
    ev_kalshi_net = ev_after_tax * (1 - KALSHI_FEE)
    money_market_return = (1 + INTEREST_RATE) ** (days_to_resolve / 365) - 1
    
    decision = "Invest" if ev_kalshi_net > money_market_return else "Don't Invest"
    
    return {
        "Expected Value (EV)": ev,
        "After-Tax EV": ev_after_tax,
        "Net EV after Kalshi Fee": ev_kalshi_net,
        "Money Market Return": money_market_return,
        "Decision": decision
    }
