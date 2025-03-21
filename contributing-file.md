# Contributing to Kalshi Investment Analyzer

Thank you for considering contributing to this project! Here are some guidelines to help you get started.

## Code of Conduct

- Be respectful
- Provide constructive feedback
- Focus on the project goals

## How Can I Contribute?

### Reporting Bugs

- Use the GitHub issue tracker
- Include detailed steps to reproduce the bug
- Specify your environment (OS, Python version, etc.)
- Include any error messages or screenshots

### Suggesting Enhancements

- Use the GitHub issue tracker
- Clearly describe the enhancement and its benefits
- Consider the project scope and goals

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests if available
5. Commit your changes (`git commit -m 'Add feature'`)
6. Push to your branch (`git push origin feature/your-feature`)
7. Create a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/kalshi-investment-analyzer.git
cd kalshi-investment-analyzer

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-api.txt  # If working on API features
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused on a single responsibility

## Testing

- Add appropriate tests for new features
- Ensure all tests pass before submitting a PR

## Feature Roadmap

Future enhancements may include:
- Extended market analysis tools
- Portfolio optimization
- Historical performance tracking
- Web interface
