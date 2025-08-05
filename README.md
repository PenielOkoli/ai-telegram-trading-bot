# 🤖 AI Telegram Trading Bot

**Autonomous cryptocurrency trading bot that monitors Telegram channels, uses GPT-4 AI to detect trading signals with 70%+ confidence, and automatically executes trades on Bybit without manual confirmation.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Bybit API](https://img.shields.io/badge/Exchange-Bybit-orange.svg)](https://www.bybit.com/)
[![OpenAI GPT-4](https://img.shields.io/badge/AI-GPT--4-green.svg)](https://openai.com/)

## ⚡ Quick Demo

```
📡 Monitoring @crypto_signals_pro...
🧠 AI Analysis: "LONG BTCUSDT" → 87% confidence
⚡ Auto-executing: 0.05 BTC @ 10x leverage
✅ Trade successful: Order ID #1234567890
💰 Position: +$127.50 (2.1% gain)
```

## 🎯 Key Features

- **🧠 AI-Powered Signal Detection**: GPT-4 analyzes messages with confidence scoring
- **📡 Real-time Channel Monitoring**: 24/7 Telegram channel surveillance
- **⚡ Instant Auto-Trading**: Zero manual confirmation required
- **🛡️ Advanced Risk Management**: Daily limits, position sizing, emergency stops
- **📊 Multi-User Support**: Handle multiple traders simultaneously
- **🔔 Real-time Notifications**: Instant trade confirmations

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/ai-telegram-trading-bot.git
cd ai-telegram-trading-bot
```

### 2. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment
Create `.env` file:
```env
TELEGRAM_BOT_TOKEN=your_bot_token
OPENAI_API_KEY=your_openai_key
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
PHONE_NUMBER=your_phone_number
```

### 4. Run Bot
```bash
python trading_bot.py
```

## 📋 Prerequisites

### Required API Keys
- **Telegram Bot Token** - Get from [@BotFather](https://t.me/BotFather)
- **OpenAI API Key** - Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Telegram API Credentials** - Get from [my.telegram.org](https://my.telegram.org)
- **Bybit API Keys** - Get from [Bybit API Management](https://www.bybit.com/app/user/api-management)

### System Requirements
- Python 3.8+
- 1GB+ RAM
- Stable internet connection
- 24/7 server (optional but recommended)

## 🛠️ Configuration

### Add Signal Channels
```bash
# In Telegram, message your bot:
/add_channel @your_signal_channel
```

### Add Users for Auto-Trading
```bash
# Format: /add_user USER_ID API_KEY API_SECRET
/add_user 123456789 your_bybit_api_key your_bybit_secret
```

### Check Status
```bash
/status  # Shows active channels, users, and recent activity
```

## 🧠 How AI Signal Detection Works

### 1. Message Analysis
```python
# AI analyzes messages like this:
"""
LONG BTCUSDT
Entry: 42,500
TP: 44,000 | 45,500  
SL: 41,000
Leverage: 10x
"""
```

### 2. AI Response
```json
{
  "is_signal": true,
  "confidence": 87,
  "direction": "LONG",
  "symbol": "BTCUSDT",
  "entry_price": 42500,
  "take_profit": [44000, 45500],
  "stop_loss": 41000,
  "leverage": 10
}
```

### 3. Auto-Execution
- ✅ Confidence ≥ 70% → Execute trade
- ❌ Confidence < 70% → Ignore signal
- 🛡️ Risk checks before execution

## 📊 Risk Management Features

| Feature | Description | Default |
|---------|-------------|---------|
| **Confidence Threshold** | Minimum AI confidence to trade | 70% |
| **Daily Loss Limit** | Max loss per day | $100 |
| **Position Sizing** | Risk per trade | 3% of balance |
| **Max Leverage** | Maximum leverage allowed | 25x |
| **Concurrent Trades** | Max open positions | 3 |

## 🔒 Security Features

- 🔐 **API Key Encryption**: Secure credential storage
- 🚨 **Emergency Stop**: Instant trading halt
- 🧪 **Testnet Support**: Safe testing environment
- 📝 **Audit Logging**: Complete trade history
- 🔍 **Balance Verification**: Pre-trade balance checks

## 📈 Performance Monitoring

### Real-time Notifications
```
🤖 AUTO TRADE EXECUTED

📊 Signal: LONG ETHUSDT (89% confidence)
💰 Size: 2.5 ETH @ 15x leverage  
✅ Status: Filled at $2,485.50
📈 Targets: $2,580 | $2,650
🛑 Stop Loss: $2,420
⏰ Time: 2024-01-15 14:30:25
```

### Daily Reports
- 📊 Signals processed
- ✅ Successful trades
- 💰 Total PnL
- 📈 Win rate statistics

## 🚨 Important Warnings

> ⚠️ **HIGH RISK**: Cryptocurrency trading involves substantial risk of loss
> 
> ⚠️ **TEST FIRST**: Always start with testnet and small amounts
> 
> ⚠️ **MONITOR CLOSELY**: AI is not infallible - supervise the bot
> 
> ⚠️ **SECURE SETUP**: Keep API keys safe and use proper security measures

## 💰 Costs

| Service | Cost | Purpose |
|---------|------|---------|
| **OpenAI API** | ~$10-30/month | AI signal analysis |
| **VPS Hosting** | $5-20/month | 24/7 operation (optional) |
| **Bybit Trading** | 0.1% fee | Exchange trading fees |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⭐ Support

If this bot helps your trading, please:
- ⭐ Star this repository
- 🐛 Report bugs in [Issues](https://github.com/yourusername/ai-telegram-trading-bot/issues)
- 💡 Suggest features in [Discussions](https://github.com/yourusername/ai-telegram-trading-bot/discussions)

## 📞 Disclaimer

This software is for educational purposes only. Users are responsible for their own trading decisions and financial losses. The authors are not liable for any damages resulting from the use of this bot.

---

**⚡ Ready to automate your crypto trading with AI? Star the repo and get started!**
