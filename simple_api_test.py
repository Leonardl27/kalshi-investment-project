import requests

# Your API key
API_KEY = "33c949a2-8f15-482a-bf09-ba226eac4698"

# Kalshi API base URL (demo environment)
BASE_URL = "https://demo-api.kalshi.co"

def test_connection():
    """Test if we can connect to Kalshi API"""
    # Try multiple endpoints that might work
    endpoints = [
        "/trade-api/v2/markets"         # correct variant
    ]
    
    print(f"Testing Kalshi API connection with multiple endpoints...")
    print(f"API Key: {API_KEY[:5]}...{API_KEY[-5:]}")
    
    # Headers for the request
    headers = {
        "KALSHI-API-KEY": API_KEY
    }
    
    success = False
    
    for endpoint in endpoints:
        url = BASE_URL + endpoint
        print(f"\nTrying URL: {url}")
        
        try:
            # Make the request
            response = requests.get(url, headers=headers)
            
            # Check if request was successful
            if response.status_code == 200:
                print(f"✅ SUCCESS! Connected to Kalshi API")
                try:
                    print(f"Response: {response.json()}")
                except:
                    print(f"Response: {response.text[:200]}")
                success = True
                break
            else:
                print(f"❌ Status code: {response.status_code}")
                print(f"Response: {response.text[:100]}")
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
    
    if not success:
        print("\n❌ Failed to connect to any Kalshi API endpoint")
        print("Possible reasons:")
        print("1. API key is invalid")
        print("2. Endpoints have changed")
        print("3. Additional authentication is required")
        print("4. Network or firewall issues")
    
    return success

if __name__ == "__main__":
    test_connection()
