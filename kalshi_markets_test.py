"""
Kalshi API Markets Test - Limited to 5 markets
"""

import requests
import json

# Your API key
API_KEY = "33c949a2-8f15-482a-bf09-ba226eac4698"

# Kalshi API base URL (demo environment)
BASE_URL = "https://demo-api.kalshi.co"

def fetch_markets(limit=5):
    """
    Fetch markets from Kalshi API and display limited results
    
    Args:
        limit: Maximum number of markets to show
    """
    endpoint = "/trade-api/v2/markets"
    url = BASE_URL + endpoint
    
    # Headers for the request
    headers = {
        "KALSHI-API-KEY": API_KEY
    }
    
    print(f"Fetching markets from Kalshi API...")
    print(f"URL: {url}")
    
    try:
        # Make the request
        response = requests.get(url, headers=headers)
        
        # Check if request was successful
        if response.status_code == 200:
            print(f"✅ SUCCESS! Connected to Kalshi API\n")
            
            # Parse the JSON response
            data = response.json()
            
            # Extract markets
            markets = data.get("markets", [])
            
            if markets:
                # Limit to specified number
                limited_markets = markets[:limit]
                
                # Display the markets
                print(f"===== Found {len(markets)} markets, showing {len(limited_markets)} =====\n")
                
                for i, market in enumerate(limited_markets, 1):
                    print(f"Market #{i}: {market.get('title', 'Untitled')}")
                    print(f"  ID: {market.get('id', 'Unknown')}")
                    print(f"  Subtitle: {market.get('subtitle', 'None')}")
                    
                    # Format close date if available
                    close_date = market.get('close_date', 'Unknown')
                    print(f"  Close Date: {close_date}")
                    
                    # Show current yes price if available
                    yes_bid = market.get('yes_bid', None)
                    yes_ask = market.get('yes_ask', None)
                    
                    if yes_bid is not None:
                        print(f"  Current Yes Bid: {yes_bid:.2f}")
                    if yes_ask is not None:
                        print(f"  Current Yes Ask: {yes_ask:.2f}")
                    
                    print()
                
                print(f"To see more details about a specific market, use:")
                print(f"GET {BASE_URL}/trade-api/v2/markets/MARKET_ID")
                
            else:
                print("No markets found in the response.")
                
        else:
            print(f"❌ Failed. Status code: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

if __name__ == "__main__":
    fetch_markets(5)  # Show only 5 markets
