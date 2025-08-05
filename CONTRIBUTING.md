# Contributing to AI Telegram Trading Bot

Thank you for your interest in contributing to this project! This guide will help you get started.

## 🤝 How to Contribute

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When creating a bug report, include:

- **Clear description** of the issue
- **Steps to reproduce** the behavior
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, etc.)
- **Log files** if applicable
- **Screenshots** if relevant

### Suggesting Features

Feature suggestions are welcome! Please:

- **Check existing issues** to avoid duplicates
- **Provide clear use case** for the feature
- **Explain the benefits** to users
- **Consider implementation complexity**

### Code Contributions

#### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-telegram-trading-bot.git
   cd ai-telegram-trading-bot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If you create this for dev tools
   ```

4. **Set up environment**
   ```bash
   cp .env.example .env
   # Fill in your test credentials
   ```

#### Development Guidelines

**Code Style:**
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and small
- Use type hints where possible

**Example:**
```python
async def analyze_message(self, message_text: str, channel_name: str = "") -> Optional[TradingSignal]:
    """
    Analyze message text for trading signals using AI.
    
    Args:
        message_text: The message content to analyze
        channel_name: Name of the source channel (optional)
        
    Returns:
        TradingSignal object if valid signal found, None otherwise
        
    Raises:
        APIError: If OpenAI API call fails
    """
```

**Testing:**
- Write tests for new features
- Ensure existing tests pass
- Test with both testnet and small real amounts
- Include edge cases in tests

**Security:**
- Never commit API keys or credentials
- Use environment variables for sensitive data
- Validate all user inputs
- Follow security best practices

#### Pull Request Process

1. **Create feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make your changes**
   - Write clean, documented code
   - Add tests if applicable
   - Update documentation

3. **Test thoroughly**
   ```bash
   python -m pytest  # If tests exist
   python -c "import trading_bot; print('Import successful')"
   ```

4. **Commit with clear messages**
   ```bash
   git commit -m "feat: add support for multiple take profit levels
   
   - Enhanced TradingSignal class to support multiple TP levels
   - Updated AI prompt to extract multiple targets
   - Added tests for multi-TP signal parsing"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/amazing-feature
   ```

6. **Create Pull Request**
   - Use clear title and description
   - Reference related issues
   - Include testing information
   - Add screenshots if UI changes

#### Commit Message Format

Use conventional commits format:

```
type(scope): short description

Longer description if needed

- List of changes
- Another change
- Fixes #123
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

## 📋 Development Areas

### High Priority
- **Enhanced AI prompts** for better signal detection
- **Additional exchange support** (Binance, OKX, etc.)
- **Improved risk management** algorithms
- **Better error handling** and recovery
- **Performance optimizations**

### Medium Priority
- **Web dashboard** for monitoring
- **Database integration** for trade history
- **Advanced analytics** and reporting
- **Mobile app integration**
- **Backtesting framework**

### Low Priority
- **Social trading features**
- **Copy trading functionality**
- **Advanced charting integration**
- **Machine learning improvements**

## 🧪 Testing Guidelines

### Manual Testing Checklist

**Before submitting PR:**
- [ ] Bot starts without errors
- [ ] Environment variables load correctly
- [ ] Telegram authentication works
- [ ] Channel monitoring functions
- [ ] AI signal detection works
- [ ] Trade execution succeeds (testnet)
- [ ] Error handling works properly
- [ ] Logging is appropriate

**Test Cases to Cover:**
- Valid trading signals (various formats)
- Invalid/incomplete signals
- Network connectivity issues
- API rate limiting
- Insufficient balance scenarios
- Emergency stop functionality

### Automated Testing

If you're adding tests, use pytest:

```python
import pytest
from trading_bot import SignalParser, TradingSignal

class TestSignalParser:
    def test_parse_long_signal(self):
        parser = SignalParser()
        signal_text = "LONG BTCUSDT\nEntry: 42000\nTP: 44000\nSL: 40000"
        
        # Mock AI response
        with patch('openai.ChatCompletion.acreate') as mock_ai:
            mock_ai.return_value.choices[0].message.content = '''
            {
                "is_signal": true,
                "confidence": 85,
                "direction": "LONG",
                "symbol": "BTCUSDT"
            }
            '''
            
            signal = parser.parse_signal(signal_text)
            assert signal.direction == "LONG"
            assert signal.symbol == "BTCUSDT"
```

## 📚 Documentation

### Code Documentation
- Add docstrings to all public functions
- Include parameter types and return values
- Provide usage examples
- Document exceptions that might be raised

### User Documentation
- Update README.md for new features
- Add configuration examples
- Include troubleshooting steps
- Provide migration guides for breaking changes

## 🚨 Security Considerations

### Security Review Required For:
- API key handling changes
- Authentication modifications
- Network request modifications
- File system operations
- Database operations
- Cryptographic functions

### Security Best Practices:
- Never log sensitive information
- Validate all external inputs
- Use secure random number generation
- Implement proper error handling
- Follow principle of least privilege
- Regular dependency updates

## 📞 Getting Help

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Code Reviews**: PR feedback and suggestions

### Before Asking for Help
1. Check existing documentation
2. Search closed issues
3. Try reproducing with minimal example
4. Include relevant error messages and logs

## 🎉 Recognition

Contributors will be:
- Added to the contributors list
- Mentioned in release notes for significant contributions
- Given credit in documentation

Thank you for helping make this project better! 🚀
