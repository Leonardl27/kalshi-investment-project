"""
Kalshi Prediction Market Investment Analyzer

A tool for analyzing betting opportunities on Kalshi prediction markets,
with comprehensive evaluation of expected returns, risk assessment,
and comparison against alternative investments.
"""

import argparse
import base64
import datetime
import json
import logging
import os
import sys
from typing import Dict, Any, Tuple, Optional, Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from dotenv import load_dotenv


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kalshi_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("kalshi_analyzer")


class KalshiAPI:
    """Handles communication with the Kalshi API."""

    def __init__(self, api_key: str, private_key_path: str, base_url: str = 'https://demo-api.kalshi.co'):
        """
        Initialize the Kalshi API client.
        
        Args:
            api_key: Your Kalshi API key
            private_key_path: Path to your RSA private key file
            base_url: Kalshi API base URL (defaults to demo environment)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.private_key = self._load_private_key(private_key_path)
        logger.info(f"Initialized Kalshi API client with base URL: {base_url}")

    @staticmethod
    def _load_private_key(file_path: str) -> rsa.RSAPrivateKey:
        """
        Load an RSA private key from a PEM file.
        
        Args:
            file_path: Path to the private key file
            
        Returns:
            An RSA private key object
            
        Raises:
            FileNotFoundError: If the key file doesn't exist
            ValueError: If the key file is invalid
        """
        try:
            with open(file_path, "rb") as key_file:
                private_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=None,
                    backend=default_backend()
                )
            return private_key
        except FileNotFoundError:
            logger.error(f"Private key file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load private key: {str(e)}")
            raise ValueError(f"Invalid private key file: {str(e)}")

    def _sign_request(self, method: str, path: str) -> Tuple[str, int]:
        """
        Create a signature for API request authentication.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: API endpoint path
            
        Returns:
            Tuple of (base64-encoded signature, timestamp in milliseconds)
        """
        timestamp_ms = int(datetime.datetime.now().timestamp() * 1000)
        msg_string = str(timestamp_ms) + method + path
        
        try:
            signature = self.private_key.sign(
                msg_string.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH
                ),
                hashes.SHA256()
            )
            return base64.b64encode(signature).decode('utf-8'), timestamp_ms
        except Exception as e:
            logger.error(f"Failed to sign request: {str(e)}")
            raise ValueError(f"RSA signature creation failed: {str(e)}")

    def make_request(self, method: str, path: str, data: Dict = None) -> Dict:
        """
        Make an authenticated request to the Kalshi API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: API endpoint path
            data: Request payload for POST requests
            
        Returns:
            JSON response from the API
            
        Raises:
            requests.RequestException: If the request fails
            ValueError: If the response is not valid JSON
        """
        signature, timestamp = self._sign_request(method, path)
        
        headers = {
            'KALSHI-ACCESS-KEY': self.api_key,
            'KALSHI-ACCESS-SIGNATURE': signature,
            'KALSHI-ACCESS-TIMESTAMP': str(timestamp),
            'Content-Type': 'application/json'
        }
        
        url = self.base_url + path
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response content: {e.response.text}")
            raise
        except ValueError as e:
            logger.error(f"Failed to parse API response: {str(e)}")
            raise

    def get_market(self, market_id: str) -> Dict:
        """
        Get details for a specific market.
        
        Args:
            market_id: The Kalshi market ID
            
        Returns:
            Market details
        """
        path = f"/trade-api/v2/markets/{market_id}"
        return self.make_request("GET", path)

    def get_markets(self, status: str = "open", limit: int = 100) -> Dict:
        """
        Get a list of markets matching the specified criteria.
        
        Args:
            status: Market status (open, closed, etc.)
            limit: Maximum number of results
            
        Returns:
            List of markets
        """
        path = f"/trade-api/v2/markets?status={status}&limit={limit}"
        return self.make_request("GET", path)
    
    def get_order_book(self, market_id: str) -> Dict:
        """
        Get the order book for a market.
        
        Args:
            market_id: The Kalshi market ID
            
        Returns:
            Order book data
        """
        path = f"/trade-api/v2/markets/{market_id}/order_book"
        return self.make_request("GET", path)
    
    def get_portfolio_balance(self) -> Dict:
        """
        Get the current portfolio balance.
        
        Returns:
            Portfolio balance information
        """
        path = "/trade-api/v2/portfolio/balance"
        return self.make_request("GET", path)


class KalshiInvestmentAnalyzer:
    """Analyzes investment opportunities in Kalshi prediction markets."""
    
    def __init__(self, config: Dict[str, float]):
        """
        Initialize the investment analyzer.
        
        Args:
            config: Configuration parameters including tax_rate, kalshi_fee, interest_rate
        """
        self.tax_rate = config.get("tax_rate", 0.25)
        self.kalshi_fee = config.get("kalshi_fee", 0.05)
        self.interest_rate = config.get("interest_rate", 0.05)
        self.risk_aversion = config.get("risk_aversion", 1.0)
        logger.info(f"Initialized KalshiInvestmentAnalyzer with config: {config}")
    
    def analyze_investment(self, p_your: float, p_market: float, days_to_resolve: int, 
                           bet_type: str = "no", investment_amount: float = 1.0) -> Dict[str, Any]:
        """
        Analyze an investment opportunity in a Kalshi market.
        
        Args:
            p_your: Your estimated probability of the event occurring
            p_market: Market-implied probability of the event occurring
            days_to_resolve: Number of days until market resolution
            bet_type: Type of bet ("yes" or "no")
            investment_amount: Amount to invest
            
        Returns:
            Dictionary of analysis results
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if not 0 <= p_your <= 1:
            raise ValueError(f"Your probability must be between 0 and 1, got {p_your}")
        if not 0 <= p_market <= 1:
            raise ValueError(f"Market probability must be between 0 and 1, got {p_market}")
        if days_to_resolve <= 0:
            raise ValueError(f"Days to resolve must be positive, got {days_to_resolve}")
        if bet_type not in ["yes", "no"]:
            raise ValueError(f"Bet type must be 'yes' or 'no', got {bet_type}")
        if investment_amount <= 0:
            raise ValueError(f"Investment amount must be positive, got {investment_amount}")
        
        # Calculate returns
        if bet_type == "yes":
            payout = (1 / p_market) - 1  # Profit per $1 bet for Yes
            ev = (p_your * payout) - ((1 - p_your) * 1)  # EV for betting on the event
        else:  # "no"
            payout = (1 / (1 - p_market)) - 1  # Profit per $1 bet for No
            ev = ((1 - p_your) * payout) - (p_your * 1)  # EV for betting against the event
        
        # Apply taxes and fees
        ev_after_tax = ev * (1 - self.tax_rate)
        ev_kalshi_net = ev_after_tax * (1 - self.kalshi_fee)
        
        # Calculate annual equivalent return
        annual_equivalent_return = ((1 + ev_kalshi_net) ** (365 / days_to_resolve)) - 1
        
        # Calculate opportunity cost
        money_market_return = (1 + self.interest_rate) ** (days_to_resolve / 365) - 1
        opportunity_cost = money_market_return * investment_amount
        
        # Calculate risk-adjusted return (Sharpe ratio)
        # Assuming a simplified variance calculation
        variance = p_your * (1 - p_your)
        std_dev = variance ** 0.5
        risk_adjusted_return = ev_kalshi_net / std_dev if std_dev > 0 else float('inf')
        
        # Decision and confidence level
        edge = ev_kalshi_net - money_market_return
        decision = "Invest" if edge > 0 else "Don't Invest"
        confidence = abs(edge) / 0.1  # Normalize to a 0-1 scale, assuming 10% is a "high" edge
        confidence = min(confidence, 1.0)  # Cap at 1.0
        
        # Calculate Kelly criterion optimal bet size
        if bet_type == "yes":
            win_prob = p_your
            net_odds = payout
        else:
            win_prob = 1 - p_your
            net_odds = payout
            
        kelly_fraction = (win_prob * (net_odds + 1) - 1) / net_odds if net_odds > 0 else 0
        kelly_fraction = max(0, min(kelly_fraction, 1))  # Bound between 0 and 1
        
        # Calculate expected profit
        expected_profit = investment_amount * ev_kalshi_net
        
        return {
            "Bet Type": bet_type.capitalize(),
            "Your Probability": p_your,
            "Market Probability": p_market,
            "Probability Edge": abs(p_your - p_market),
            "Days to Resolve": days_to_resolve,
            "Payout Multiplier": payout + 1,  # Total return including initial stake
            "Expected Value (EV)": ev,
            "After-Tax EV": ev_after_tax,
            "Net EV after Kalshi Fee": ev_kalshi_net,
            "Annual Equivalent Return": annual_equivalent_return,
            "Money Market Return": money_market_return,
            "Risk-Adjusted Return": risk_adjusted_return,
            "Kelly Criterion Bet Size": kelly_fraction,
            "Expected Profit": expected_profit,
            "Decision": decision,
            "Confidence": confidence
        }
    
    def compare_strategies(self, p_your: float, p_market: float, days_to_resolve: int,
                         investment_amount: float = 1000.0) -> Dict[str, Dict[str, Any]]:
        """
        Compare different betting strategies.
        
        Args:
            p_your: Your estimated probability of the event occurring
            p_market: Market-implied probability of the event occurring
            days_to_resolve: Number of days until market resolution
            investment_amount: Amount to invest
            
        Returns:
            Dictionary comparing different strategies
        """
        yes_analysis = self.analyze_investment(p_your, p_market, days_to_resolve, "yes", investment_amount)
        no_analysis = self.analyze_investment(p_your, p_market, days_to_resolve, "no", investment_amount)
        money_market = {
            "Expected Profit": investment_amount * ((1 + self.interest_rate) ** (days_to_resolve / 365) - 1),
            "Annual Equivalent Return": self.interest_rate,
            "Risk-Adjusted Return": float('inf'),  # Assuming risk-free
            "Decision": "Reference",
            "Confidence": 1.0
        }
        
        # Determine best strategy
        strategies = {
            "Yes Bet": yes_analysis,
            "No Bet": no_analysis,
            "Money Market": money_market
        }
        
        return strategies
    
    def sensitivity_analysis(self, p_your: float, p_market: float, days_to_resolve: int,
                           bet_type: str = "no", delta: float = 0.05) -> Dict[str, List[float]]:
        """
        Perform sensitivity analysis by varying your probability estimate.
        
        Args:
            p_your: Your base estimated probability
            p_market: Market-implied probability
            days_to_resolve: Days to resolution
            bet_type: Type of bet ("yes" or "no")
            delta: Range to vary probability (+/- this amount)
            
        Returns:
            Dictionary with probability variations and corresponding metrics
        """
        probabilities = []
        expected_values = []
        net_evs = []
        
        # Limit the range to valid probability values
        lower_bound = max(0, p_your - delta)
        upper_bound = min(1, p_your + delta)
        step = delta / 10
        
        for prob in np.arange(lower_bound, upper_bound + step, step):
            analysis = self.analyze_investment(prob, p_market, days_to_resolve, bet_type)
            probabilities.append(prob)
            expected_values.append(analysis["Expected Value (EV)"])
            net_evs.append(analysis["Net EV after Kalshi Fee"])
        
        return {
            "Probabilities": probabilities,
            "Expected Values": expected_values,
            "Net EVs": net_evs
        }
    
    def visualize_results(self, analysis: Dict[str, Any], title: str = "Investment Analysis") -> None:
        """
        Visualize the investment analysis results.
        
        Args:
            analysis: Analysis results from analyze_investment
            title: Title for the visualization
        """
        # Create a figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Key metrics bar chart
        metrics = ['Net EV after Kalshi Fee', 'Money Market Return', 'Annual Equivalent Return']
        values = [analysis[metric] for metric in metrics]
        colors = ['green' if v > 0 else 'red' for v in values]
        
        ax1.bar(metrics, values, color=colors)
        ax1.set_title('Key Return Metrics')
        ax1.set_ylabel('Return Rate')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        for i, v in enumerate(values):
            ax1.text(i, v + 0.001, f"{v:.2%}", ha='center')
        
        # Decision gauge chart (simplified)
        confidence = analysis['Confidence']
        decision = analysis['Decision']
        
        # Create a gauge-like representation
        if decision == "Invest":
            color = 'green'
            ax2.text(0.5, 0.5, f"INVEST\nConfidence: {confidence:.2f}", 
                    ha='center', va='center', fontsize=16, color=color)
        else:
            color = 'red'
            ax2.text(0.5, 0.5, f"DON'T INVEST\nConfidence: {confidence:.2f}", 
                    ha='center', va='center', fontsize=16, color=color)
        
        # Add a circular border
        circle = plt.Circle((0.5, 0.5), 0.4, fill=False, color=color, linewidth=3)
        ax2.add_artist(circle)
        
        # Remove axis ticks and labels
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('Investment Decision')
        
        # Add overall title and adjust layout
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        # Save the figure
        plt.savefig("kalshi_analysis.png")
        logger.info("Analysis visualization saved to kalshi_analysis.png")
        
        # Show the plot
        plt.show()
    
    def visualize_sensitivity(self, sensitivity_data: Dict[str, List[float]], 
                            p_market: float, bet_type: str) -> None:
        """
        Visualize sensitivity analysis results.
        
        Args:
            sensitivity_data: Results from sensitivity_analysis
            p_market: Market probability for reference
            bet_type: Type of bet being analyzed
        """
        plt.figure(figsize=(10, 6))
        
        probabilities = sensitivity_data["Probabilities"]
        net_evs = sensitivity_data["Net EVs"]
        
        # Plot the sensitivity curve
        plt.plot(probabilities, net_evs, 'b-', linewidth=2, label='Net EV after fees and taxes')
        
        # Add reference lines
        plt.axhline(y=0, color='r', linestyle='--', label='Break-even point')
        plt.axvline(x=p_market, color='g', linestyle='--', 
                   label=f'Market probability ({p_market:.2f})')
        
        # Add money market reference
        money_market = self.interest_rate / 365  # Daily rate
        plt.axhline(y=money_market, color='orange', linestyle=':', 
                   label=f'Money market daily return ({money_market:.4f})')
        
        # Find optimal probability for this bet type
        if bet_type == "yes":
            optimal_idx = np.argmax(net_evs)
        else:
            optimal_idx = np.argmin([abs(ev) for ev in net_evs])
        
        optimal_prob = probabilities[optimal_idx]
        optimal_ev = net_evs[optimal_idx]
        
        plt.scatter([optimal_prob], [optimal_ev], color='red', s=100, 
                   label=f'Optimal probability: {optimal_prob:.2f}')
        
        # Labels and formatting
        plt.title(f'Sensitivity Analysis for {bet_type.capitalize()} Bet')
        plt.xlabel('Your Probability Estimate')
        plt.ylabel('Net Expected Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the figure
        plt.savefig("kalshi_sensitivity.png")
        logger.info("Sensitivity analysis visualization saved to kalshi_sensitivity.png")
        
        # Show the plot
        plt.tight_layout()
        plt.show()


class KalshiAnalyzerCLI:
    """Command line interface for the Kalshi Analyzer tool."""
    
    def __init__(self):
        """Initialize the CLI with argument parser."""
        self.parser = self._setup_argparse()
        
    @staticmethod
    def _setup_argparse() -> argparse.ArgumentParser:
        """
        Set up command line argument parser.
        
        Returns:
            Configured argument parser
        """
        parser = argparse.ArgumentParser(
            description="Kalshi Prediction Market Investment Analyzer",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Add subparsers for different commands
        subparsers = parser.add_subparsers(dest="command", help="Command to execute")
        
        # Analyze command
        analyze_parser = subparsers.add_parser("analyze", help="Analyze a single investment opportunity")
        analyze_parser.add_argument("--your-prob", type=float, required=True, 
                                  help="Your estimated probability (0-1)")
        analyze_parser.add_argument("--market-prob", type=float, required=True, 
                                  help="Market-implied probability (0-1)")
        analyze_parser.add_argument("--days", type=int, required=True, 
                                  help="Days until market resolution")
        analyze_parser.add_argument("--bet-type", choices=["yes", "no"], default="no", 
                                  help="Type of bet to analyze")
        analyze_parser.add_argument("--amount", type=float, default=1000, 
                                  help="Investment amount in dollars")
        analyze_parser.add_argument("--visualize", action="store_true", 
                                  help="Generate visualization of results")
        
        # Compare command
        compare_parser = subparsers.add_parser("compare", help="Compare multiple betting strategies")
        compare_parser.add_argument("--your-prob", type=float, required=True, 
                                  help="Your estimated probability (0-1)")
        compare_parser.add_argument("--market-prob", type=float, required=True, 
                                  help="Market-implied probability (0-1)")
        compare_parser.add_argument("--days", type=int, required=True, 
                                  help="Days until market resolution")
        compare_parser.add_argument("--amount", type=float, default=1000, 
                                  help="Investment amount in dollars")
        
        # Sensitivity command
        sensitivity_parser = subparsers.add_parser("sensitivity", 
                                                help="Perform sensitivity analysis")
        sensitivity_parser.add_argument("--your-prob", type=float, required=True, 
                                      help="Your estimated probability (0-1)")
        sensitivity_parser.add_argument("--market-prob", type=float, required=True, 
                                      help="Market-implied probability (0-1)")
        sensitivity_parser.add_argument("--days", type=int, required=True, 
                                      help="Days until market resolution")
        sensitivity_parser.add_argument("--bet-type", choices=["yes", "no"], default="no", 
                                      help="Type of bet to analyze")
        sensitivity_parser.add_argument("--delta", type=float, default=0.1, 
                                      help="Range to vary probability (+/- this amount)")
        
        # Market lookup command
        market_parser = subparsers.add_parser("market", help="Look up market information")
        market_parser.add_argument("--market-id", type=str, required=True, 
                                 help="Kalshi market ID to look up")
        
        # Global options
        parser.add_argument("--config", type=str, default="config.json", 
                          help="Path to configuration file")
        parser.add_argument("--api-key", type=str, help="Kalshi API key (overrides config)")
        parser.add_argument("--private-key", type=str, 
                          help="Path to private key file (overrides config)")
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        
        return parser
    
    def parse_args(self) -> argparse.Namespace:
        """
        Parse command line arguments.
        
        Returns:
            Parsed arguments
        """
        return self.parser.parse_args()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If the config file is invalid JSON
        """
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {str(e)}")
            raise ValueError(f"Invalid configuration file: {str(e)}")
    
    def run(self) -> None:
        """Run the CLI application."""
        args = self.parse_args()
        
        # Set up logging level
        if args.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")
        
        # Load configuration
        try:
            config = self.load_config(args.config)
        except (FileNotFoundError, ValueError):
            logger.info("Creating default configuration")
            config = {
                "tax_rate": 0.25,
                "kalshi_fee": 0.05,
                "interest_rate": 0.05,
                "risk_aversion": 1.0
            }
        
        # Check for environment variables
        load_dotenv()
        env_api_key = os.getenv("KALSHI_API_KEY")
        env_private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        
        # Determine API credentials (priority: CLI args > env vars > config file)
        api_key = args.api_key or env_api_key or config.get("api_key")
        private_key_path = args.private_key or env_private_key_path or config.get("private_key_path")
        
        # Initialize components
        analyzer = KalshiInvestmentAnalyzer(config)
        
        # Execute the selected command
        if args.command == "analyze":
            result = analyzer.analyze_investment(
                p_your=args.your_prob,
                p_market=args.market_prob,
                days_to_resolve=args.days,
                bet_type=args.bet_type,
                investment_amount=args.amount
            )
            
            # Display results
            print("\n===== Investment Analysis =====")
            for key, value in result.items():
                if isinstance(value, float):
                    if key in ["Your Probability", "Market Probability", "Probability Edge", "Confidence"]:
                        print(f"{key}: {value:.2f}")
                    elif "Return" in key or "EV" in key:
                        print(f"{key}: {value:.2%}")
                    else:
                        print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
            
            # Generate visualization if requested
            if args.visualize:
                analyzer.visualize_results(result)
        
        elif args.command == "compare":
            strategies = analyzer.compare_strategies(
                p_your=args.your_prob,
                p_market=args.market_prob,
                days_to_resolve=args.days,
                investment_amount=args.amount
            )
            
            # Display results as a comparison table
            print("\n===== Strategy Comparison =====")
            
            # Create a pandas DataFrame for nice display
            data = {
                "Metric": ["Expected Profit", "Annual Equivalent Return", 
                         "Risk-Adjusted Return", "Decision", "Confidence"]
            }
            
            for strategy, analysis in strategies.items():
                data[strategy] = [
                    f"${analysis.get('Expected Profit', 0):.2f}",
                    f"{analysis.get('Annual Equivalent Return', 0):.2%}",
                    f"{analysis.get('Risk-Adjusted Return', 0):.2f}",
                    analysis.get('Decision', 'N/A'),
                    f"{analysis.get('Confidence', 0):.2f}"
                ]
            
            df = pd.DataFrame(data)
            print(df.to_string(index=False))
            
            # Determine best strategy
            best_strategy = max(strategies.items(), 
                              key=lambda x: x[1].get('Expected Profit', 0))
            print(f"\nRecommended Strategy: {best_strategy[0]}")
        
        elif args.command == "sensitivity":
            sensitivity_data = analyzer.sensitivity_analysis(
                p_your=args.your_prob,
                p_market=args.market_prob,
                days_to_resolve=args.days,
                bet_type=args.bet_type,
                delta=args.delta
            )
            
            # Generate visualization
            analyzer.visualize_sensitivity(sensitivity_data, args.market_prob, args.bet_type)
        
        elif args.command == "market":
            # Ensure we have API credentials
            if not api_key or not private_key_path:
                logger.error("API key and private key path are required for market lookup")
                sys.exit(1)
            
            # Initialize API client
            api_client = KalshiAPI(api_key, private_key_path)
            
            try:
                market_data = api_client.get_market(args.market_id)
                
                # Extract and display relevant information
                print(f"\n===== Market: {market_data.get('market', {}).get('title', 'Unknown')} =====")
                market = market_data.get('market', {})
                
                # Market details
                print(f"ID: {market.get('id')}")
                print(f"Title: {market.get('title')}")
                print(f"Subtitle: {market.get('subtitle')}")
                print(f"Close Date: {market.get('close_date')}")
                print(f"Settlement Date: {market.get('settlement_date')}")
                
                # Current prices
                yes_price = market.get('yes_bid', 0)
                no_price = 1 - market.get('yes_ask', 1)
                print(f"Current Yes Price: {yes_price:.2f}")
                print(f"Current No Price: {no_price:.2f}")
                print(f"Implied Probability: {yes_price:.2%}")
                
                # Calculate days to resolution
                if market.get('settlement_date'):
                    settlement_date = datetime.datetime.fromisoformat(
                        market.get('settlement_date').replace('Z', '+00:00')
                    )
                    now = datetime.datetime.now(datetime.timezone.utc)
                    days_to_resolve = (settlement_date - now).days
                    print(f"Days to Resolution: {days_to_resolve}")
                
                # Suggest next steps
                print("\nSuggested Next Steps:")
                print(f"  analyze --your-prob 0.XX --market-prob {yes_price:.2f} --days {days_to_resolve}")
                
            except Exception as e:
                logger.error(f"Failed to retrieve market data: {str(e)}")
                sys.exit(1)
        
        else:
            self.parser.print_help()


def main():
    """Main entry point for the application."""
    try:
        cli = KalshiAnalyzerCLI()
        cli.run()
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        if logger.level == logging.DEBUG:
            logger.exception("Detailed traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
