# Kalshi Investment Analyzer

A comprehensive tool for analyzing investment opportunities in Kalshi prediction markets with quantitative decision-making capabilities.

## Features

- **Investment Analysis**: Calculate expected value, risk-adjusted returns, and optimal position sizing
- **Strategy Comparison**: Compare "Yes" vs "No" bets against alternative investments
- **Decision Support**: Get clear recommendations based on your probability estimates vs market prices
- **Kelly Criterion**: Determine optimal bet sizes based on your edge
- **Tax & Fee Modeling**: Account for taxes and platform fees in your decision-making

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/kalshi-investment-analyzer.git
cd kalshi-investment-analyzer

# Install dependencies
pip install -r requirements.txt
```

### API Integration (Optional)

For API integration features, install additional dependencies:

```bash
pip install -r requirements-api.txt
```

## Usage

### Basic Analysis

```bash
python kalshi_analyzer.py analyze --your-prob 0.30 --market-prob 0.39 --days 45 --amount 1000
```

### Compare Strategies

```bash
python kalshi_analyzer.py compare --your-prob 0.30 --market-prob 0.39 --days 45 --amount 1000
```

### Configuration

Create a `config.json` file in the project root:

```json
{
  "tax_rate": 0.25,
  "kalshi_fee": 0.05,
  "interest_rate": 0.05,
  "risk_aversion": 1.0
}
```

## Theory

This tool helps determine if a market offers positive expected value by comparing:

1. Your probability estimate (P_your)
2. The market-implied probability (P_market)
3. Alternative investment returns (e.g., risk-free rate)

The core calculation for a "No" bet is:
```
EV = ((1 - P_your) * Payout) - (P_your * 1)
```

Where `Payout = (1 / (1 - P_market)) - 1`

## License

MIT

## Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md).
