import asyncio
import re
import logging
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import json
import os
from dataclasses import dataclass
import time

# Required imports (install via pip)
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from pybit.unified_trading import HTTP
import yaml
import openai
from telethon import TelegramClient, events
from telethon.tl.types import Channel

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Enhanced data class to store parsed trading signals with confidence scores"""
    direction: str  # LONG or SHORT
    symbol: str
    order_type: str  # MARKET or LIMIT
    entry_price: Optional[float] = None
    take_profit: List[float] = None  # Multiple TP levels
    stop_loss: float = 0
    leverage: int = 10
    risk_percentage: float = 5.0
    confidence_score: float = 0.0  # AI confidence in signal validity
    original_message: str = ""
    timestamp: datetime = None

    def __post_init__(self):
        if self.take_profit is None:
            self.take_profit = []
        if self.timestamp is None:
            self.timestamp = datetime.now()

class AISignalAnalyzer:
    """AI-powered signal analysis and extraction"""
    
    def __init__(self, openai_api_key: str):
        openai.api_key = openai_api_key
        self.signal_patterns = [
            # Various signal formats the AI should recognize
            r'(LONG|SHORT|BUY|SELL)',
            r'([A-Z0-9]+)(USDT|/USDT)',
            r'(MARKET|LIMIT)',
            r'(TP|TAKE\s*PROFIT|TARGET)',
            r'(SL|STOP\s*LOSS)',
            r'(LEVERAGE|LEV)',
        ]
    
    async def analyze_message(self, message_text: str, channel_name: str = "") -> Optional[TradingSignal]:
        """
        Use AI to analyze message and extract trading signal with confidence scoring
        """
        try:
            # First, quick pattern check to avoid unnecessary API calls
            if not self._contains_trading_keywords(message_text):
                return None
            
            # Prepare prompt for AI analysis
            prompt = self._create_analysis_prompt(message_text, channel_name)
            
            # Get AI analysis
            response = await self._get_ai_response(prompt)
            
            if not response:
                return None
            
            # Parse AI response and create signal
            signal = self._parse_ai_response(response, message_text)
            
            # Validate signal quality
            if signal and self._validate_signal(signal):
                return signal
                
            return None
            
        except Exception as e:
            logger.error(f"Error in AI signal analysis: {e}")
            return None
    
    def _contains_trading_keywords(self, text: str) -> bool:
        """Quick check for trading-related keywords"""
        keywords = [
            'LONG', 'SHORT', 'BUY', 'SELL', 'USDT', 'BTC', 'ETH',
            'TP', 'SL', 'LEVERAGE', 'ENTRY', 'TARGET', 'STOP',
            'MARKET', 'LIMIT', 'SIGNAL', 'TRADE'
        ]
        text_upper = text.upper()
        return any(keyword in text_upper for keyword in keywords)
    
    def _create_analysis_prompt(self, message: str, channel: str) -> str:
        """Create prompt for AI signal analysis"""
        return f"""
Analyze this trading message and extract trading signal information. 
Channel: {channel}
Message: "{message}"

Your task is to:
1. Determine if this is a valid trading signal
2. Extract all relevant trading information
3. Provide confidence score (0-100)

Expected signal format to extract:
- Direction: LONG/SHORT/BUY/SELL
- Symbol: cryptocurrency pair (e.g., BTCUSDT, ETHUSDT)
- Entry type: MARKET/LIMIT
- Entry price: if limit order
- Take Profit levels: one or multiple targets
- Stop Loss: risk management level
- Leverage: if specified
- Risk percentage: if mentioned

Response format (JSON):
{{
    "is_signal": true/false,
    "confidence": 0-100,
    "direction": "LONG/SHORT",
    "symbol": "SYMBOL",
    "order_type": "MARKET/LIMIT",
    "entry_price": null or number,
    "take_profit": [numbers],
    "stop_loss": number or null,
    "leverage": number or 10,
    "reasoning": "why you think this is/isn't a signal"
}}

Be strict about signal validity. Only mark as signal if clearly contains trading instruction.
"""

    async def _get_ai_response(self, prompt: str) -> Optional[str]:
        """Get response from OpenAI"""
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert cryptocurrency trading signal analyzer. Analyze messages for valid trading signals with high accuracy."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1  # Low temperature for consistent analysis
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None
    
    def _parse_ai_response(self, ai_response: str, original_message: str) -> Optional[TradingSignal]:
        """Parse AI response into TradingSignal object"""
        try:
            # Extract JSON from AI response
            import json
            
            # Try to find JSON in the response
            json_start = ai_response.find('{')
            json_end = ai_response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                return None
            
            json_str = ai_response[json_start:json_end]
            analysis = json.loads(json_str)
            
            # Check if AI considers this a valid signal
            if not analysis.get('is_signal', False):
                return None
            
            # Extract signal data
            signal = TradingSignal(
                direction=analysis.get('direction', '').upper(),
                symbol=self._normalize_symbol(analysis.get('symbol', '')),
                order_type=analysis.get('order_type', 'MARKET').upper(),
                entry_price=analysis.get('entry_price'),
                take_profit=analysis.get('take_profit', []),
                stop_loss=analysis.get('stop_loss', 0),
                leverage=analysis.get('leverage', 10),
                confidence_score=analysis.get('confidence', 0),
                original_message=original_message
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return None
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format for Bybit"""
        if not symbol:
            return ""
        
        symbol = symbol.upper().replace('/', '')
        if not symbol.endswith('USDT'):
            symbol += 'USDT'
        
        return symbol
    
    def _validate_signal(self, signal: TradingSignal) -> bool:
        """Validate signal quality and completeness"""
        # Check required fields
        if not signal.direction or signal.direction not in ['LONG', 'SHORT', 'BUY', 'SELL']:
            return False
        
        if not signal.symbol or 'USDT' not in signal.symbol:
            return False
        
        # Confidence threshold
        if signal.confidence_score < 70:  # Require high confidence
            return False
        
        # Basic sanity checks
        if signal.leverage and (signal.leverage < 1 or signal.leverage > 125):
            return False
        
        return True

class ChannelMonitor:
    """Monitors specific Telegram channels for trading signals"""
    
    def __init__(self, api_id: int, api_hash: str, phone: str, signal_analyzer: AISignalAnalyzer):
        self.client = TelegramClient('trading_session', api_id, api_hash)
        self.phone = phone
        self.signal_analyzer = signal_analyzer
        self.monitored_channels = []
        self.signal_callback = None
        
    async def initialize(self):
        """Initialize Telegram client"""
        try:
            await self.client.start(phone=self.phone)
            logger.info("Channel monitor initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize channel monitor: {e}")
            return False
    
    async def add_channel(self, channel_username: str) -> bool:
        """Add channel to monitoring list"""
        try:
            # Get channel entity
            channel = await self.client.get_entity(channel_username)
            
            if isinstance(channel, Channel):
                self.monitored_channels.append({
                    'entity': channel,
                    'username': channel_username,
                    'title': channel.title
                })
                
                logger.info(f"Added channel: {channel.title} (@{channel_username})")
                return True
            else:
                logger.error(f"Not a valid channel: {channel_username}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding channel {channel_username}: {e}")
            return False
    
    def set_signal_callback(self, callback):
        """Set callback function for when signals are detected"""
        self.signal_callback = callback
    
    async def start_monitoring(self):
        """Start monitoring channels for new messages"""
        @self.client.on(events.NewMessage())
        async def handle_new_message(event):
            try:
                # Check if message is from monitored channels
                channel_info = None
                for channel in self.monitored_channels:
                    if event.chat_id == channel['entity'].id:
                        channel_info = channel
                        break
                
                if not channel_info:
                    return
                
                message_text = event.message.message
                if not message_text:
                    return
                
                logger.info(f"New message from {channel_info['title']}: {message_text[:100]}...")
                
                # Analyze message with AI
                signal = await self.signal_analyzer.analyze_message(
                    message_text, 
                    channel_info['title']
                )
                
                if signal and self.signal_callback:
                    logger.info(f"Signal detected with {signal.confidence_score}% confidence")
                    await self.signal_callback(signal, channel_info)
                
            except Exception as e:
                logger.error(f"Error handling message: {e}")
        
        logger.info("Started monitoring channels...")
        await self.client.run_until_disconnected()

class EnhancedBybitTrader:
    """Enhanced Bybit trader with better risk management"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.session = HTTP(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet
        )
        self.active_positions = {}
        self.trade_history = []
        
    async def execute_signal_automatically(self, signal: TradingSignal, user_config: Dict) -> Dict:
        """Execute trading signal automatically with enhanced risk management"""
        try:
            # Pre-execution checks
            if not await self._pre_trade_checks(signal, user_config):
                return {'success': False, 'error': 'Pre-trade checks failed'}
            
            # Calculate position size with dynamic risk management
            position_size = await self._calculate_position_size(signal, user_config)
            
            if position_size <= 0:
                return {'success': False, 'error': 'Invalid position size calculated'}
            
            # Set leverage
            await self._set_leverage_safely(signal.symbol, user_config.get('leverage', signal.leverage))
            
            # Place main order
            order_result = await self._place_main_order(signal, position_size, user_config)
            
            if not order_result['success']:
                return order_result
            
            # Set up TP/SL orders
            await self._setup_exit_orders(signal, position_size, order_result['order_id'])
            
            # Record trade
            trade_record = {
                'signal': signal,
                'order_result': order_result,
                'timestamp': datetime.now(),
                'position_size': position_size
            }
            
            self.trade_history.append(trade_record)
            
            return {
                'success': True,
                'order_id': order_result['order_id'],
                'position_size': position_size,
                'confidence': signal.confidence_score,
                'message': f"Trade executed automatically with {signal.confidence_score}% confidence"
            }
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _pre_trade_checks(self, signal: TradingSignal, user_config: Dict) -> bool:
        """Comprehensive pre-trade validation"""
        try:
            # Check account balance
            balance = await self.get_account_balance()
            if not balance or not self._has_sufficient_balance(balance, user_config):
                logger.warning("Insufficient account balance")
                return False
            
            # Check if symbol is tradeable
            if not await self._is_symbol_active(signal.symbol):
                logger.warning(f"Symbol {signal.symbol} is not active/tradeable")
                return False
            
            # Check for existing positions (avoid overexposure)
            if await self._has_conflicting_position(signal):
                logger.warning(f"Conflicting position exists for {signal.symbol}")
                return False
            
            # Market hours check (some exchanges have maintenance)
            if not await self._is_market_open():
                logger.warning("Market is not open for trading")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Pre-trade check error: {e}")
            return False
    
    async def _calculate_position_size(self, signal: TradingSignal, user_config: Dict) -> float:
        """Calculate optimal position size based on risk management"""
        try:
            balance = await self.get_account_balance()
            if not balance:
                return 0
            
            # Get USDT balance
            usdt_balance = self._extract_usdt_balance(balance)
            
            # Base risk percentage
            base_risk = user_config.get('risk', 5.0)
            
            # Adjust risk based on AI confidence
            confidence_multiplier = signal.confidence_score / 100.0
            adjusted_risk = base_risk * confidence_multiplier
            
            # Apply daily loss limits
            daily_loss = self._calculate_daily_loss()
            max_daily_loss = user_config.get('max_daily_loss', usdt_balance * 0.1)
            
            if daily_loss >= max_daily_loss:
                logger.warning("Daily loss limit reached")
                return 0
            
            # Calculate position size
            risk_amount = usdt_balance * (adjusted_risk / 100)
            leverage = user_config.get('leverage', signal.leverage)
            
            # Get current price
            current_price = await self._get_current_price(signal.symbol)
            if not current_price:
                return 0
            
            # Calculate quantity
            position_value = risk_amount * leverage
            quantity = position_value / current_price
            
            # Round to appropriate decimal places
            quantity = self._round_quantity(quantity, signal.symbol)
            
            return quantity
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    async def _place_main_order(self, signal: TradingSignal, quantity: float, user_config: Dict) -> Dict:
        """Place the main trading order"""
        try:
            # Determine order side
            side = "Buy" if signal.direction in ["LONG", "BUY"] else "Sell"
            
            # Prepare order parameters
            order_params = {
                "category": "linear",
                "symbol": signal.symbol,
                "side": side,
                "orderType": signal.order_type,
                "qty": str(quantity),
                "timeInForce": "GTC"
            }
            
            # Add price for limit orders
            if signal.order_type == "LIMIT" and signal.entry_price:
                order_params["price"] = str(signal.entry_price)
            
            # Place order
            result = self.session.place_order(**order_params)
            
            if result['retCode'] == 0:
                order_id = result['result']['orderId']
                return {
                    'success': True,
                    'order_id': order_id,
                    'params': order_params
                }
            else:
                return {
                    'success': False,
                    'error': result.get('retMsg', 'Unknown error')
                }
                
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _setup_exit_orders(self, signal: TradingSignal, quantity: float, main_order_id: str):
        """Set up take profit and stop loss orders"""
        try:
            side = "Sell" if signal.direction in ["LONG", "BUY"] else "Buy"
            
            # Set up stop loss
            if signal.stop_loss > 0:
                sl_params = {
                    "category": "linear",
                    "symbol": signal.symbol,
                    "side": side,
                    "orderType": "Market",
                    "qty": str(quantity),
                    "triggerPrice": str(signal.stop_loss),
                    "triggerBy": "LastPrice"
                }
                
                self.session.place_order(**sl_params)
            
            # Set up take profit orders (potentially multiple levels)
            if signal.take_profit:
                for i, tp_price in enumerate(signal.take_profit):
                    # Divide quantity among TP levels
                    tp_quantity = quantity / len(signal.take_profit)
                    
                    tp_params = {
                        "category": "linear",
                        "symbol": signal.symbol,
                        "side": side,
                        "orderType": "Limit",
                        "qty": str(tp_quantity),
                        "price": str(tp_price),
                        "timeInForce": "GTC"
                    }
                    
                    self.session.place_order(**tp_params)
                    
        except Exception as e:
            logger.error(f"Error setting up exit orders: {e}")
    
    # Helper methods
    async def get_account_balance(self) -> Dict:
        """Get account balance"""
        try:
            result = self.session.get_wallet_balance(accountType="UNIFIED")
            return result
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return {}
    
    def _extract_usdt_balance(self, balance_data: Dict) -> float:
        """Extract USDT balance from balance response"""
        try:
            for coin in balance_data['result']['list'][0]['coin']:
                if coin['coin'] == 'USDT':
                    return float(coin['walletBalance'])
            return 0
        except:
            return 0
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            ticker = self.session.get_tickers(category="linear", symbol=symbol)
            if ticker['retCode'] == 0:
                return float(ticker['result']['list'][0]['lastPrice'])
            return None
        except:
            return None
    
    def _round_quantity(self, quantity: float, symbol: str) -> float:
        """Round quantity to appropriate decimal places"""
        # This would typically check symbol info for minimum quantity increments
        # For now, using a general approach
        if quantity < 0.001:
            return round(quantity, 6)
        elif quantity < 1:
            return round(quantity, 4)
        else:
            return round(quantity, 2)
    
    async def _set_leverage_safely(self, symbol: str, leverage: int):
        """Set leverage with error handling"""
        try:
            result = self.session.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage)
            )
            return result['retCode'] == 0
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
            return False
    
    def _has_sufficient_balance(self, balance: Dict, user_config: Dict) -> bool:
        """Check if account has sufficient balance"""
        usdt_balance = self._extract_usdt_balance(balance)
        min_balance = user_config.get('min_balance', 10)  # Minimum $10
        return usdt_balance >= min_balance
    
    async def _is_symbol_active(self, symbol: str) -> bool:
        """Check if symbol is active for trading"""
        try:
            ticker = self.session.get_tickers(category="linear", symbol=symbol)
            return ticker['retCode'] == 0
        except:
            return False
    
    async def _has_conflicting_position(self, signal: TradingSignal) -> bool:
        """Check for conflicting positions"""
        try:
            positions = self.session.get_positions(category="linear", symbol=signal.symbol)
            if positions['retCode'] == 0 and positions['result']['list']:
                for pos in positions['result']['list']:
                    if float(pos['size']) > 0:  # Has active position
                        return True
            return False
        except:
            return False
    
    async def _is_market_open(self) -> bool:
        """Check if market is open (crypto markets are always open, but check for maintenance)"""
        # For crypto, markets are 24/7, but you might want to check for exchange maintenance
        return True
    
    def _calculate_daily_loss(self) -> float:
        """Calculate today's total loss"""
        today = datetime.now().date()
        daily_loss = 0
        
        for trade in self.trade_history:
            if trade['timestamp'].date() == today:
                # This would need to be calculated based on actual PnL
                # For now, assuming we track this separately
                pass
        
        return daily_loss

class AutoTradingBot:
    """Main auto-trading bot with AI signal detection"""
    
    def __init__(self, telegram_token: str, openai_api_key: str, telegram_api_id: int, 
                 telegram_api_hash: str, phone_number: str):
        self.config_manager = ConfigManager()
        self.signal_analyzer = AISignalAnalyzer(openai_api_key)
        self.channel_monitor = ChannelMonitor(telegram_api_id, telegram_api_hash, phone_number, self.signal_analyzer)
        
        # Set up signal callback
        self.channel_monitor.set_signal_callback(self.handle_detected_signal)
        
        # Telegram bot for admin commands
        self.admin_app = Application.builder().token(telegram_token).build()
        self.setup_admin_handlers()
        
        # Active traders
        self.active_traders = {}
        
    def setup_admin_handlers(self):
        """Setup admin command handlers"""
        self.admin_app.add_handler(CommandHandler("start", self.admin_start))
        self.admin_app.add_handler(CommandHandler("add_channel", self.add_channel_command))
        self.admin_app.add_handler(CommandHandler("add_user", self.add_user_command))
        self.admin_app.add_handler(CommandHandler("status", self.status_command))
        self.admin_app.add_handler(CommandHandler("stop_trading", self.stop_trading_command))
        
    async def handle_detected_signal(self, signal: TradingSignal, channel_info: Dict):
        """Handle AI-detected trading signal"""
        try:
            logger.info(f"Processing signal: {signal.direction} {signal.symbol} (Confidence: {signal.confidence_score}%)")
            
            # Execute trades for all active users
            for user_id, user_config in self.config_manager.config.get('users', {}).items():
                if not user_config.get('auto_trade', False):
                    continue
                
                if user_id not in self.active_traders:
                    self.active_traders[user_id] = EnhancedBybitTrader(
                        user_config['api_key'],
                        user_config['api_secret'],
                        user_config.get('testnet', False)
                    )
                
                trader = self.active_traders[user_id]
                
                # Execute trade automatically
                result = await trader.execute_signal_automatically(signal, user_config)
                
                # Log result
                if result['success']:
                    logger.info(f"Trade executed for user {user_id}: {result['message']}")
                    # Optionally send notification to user
                    await self.send_trade_notification(user_id, signal, result)
                else:
                    logger.error(f"Trade failed for user {user_id}: {result['error']}")
                    
        except Exception as e:
            logger.error(f"Error handling detected signal: {e}")
    
    async def send_trade_notification(self, user_id: str, signal: TradingSignal, result: Dict):
        """Send trade notification to user"""
        try:
            message = f"""
🤖 **AUTO TRADE EXECUTED**

📊 **Signal Details:**
Direction: {signal.direction}
Symbol: {signal.symbol}
Confidence: {signal.confidence_score}%

💰 **Trade Details:**
Order ID: {result.get('order_id', 'N/A')}
Position Size: {result.get('position_size', 'N/A')}
Status: ✅ Success

⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            # Send to user (you'll need to implement this based on your notification preference)
            await self.admin_app.bot.send_message(chat_id=int(user_id), text=message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    # Admin command handlers
    async def admin_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Admin start command"""
        await update.message.reply_text("""
🤖 **Auto-Trading Bot Admin Panel**

Commands:
/add_channel <username> - Add channel to monitor
/add_user <user_id> <api_key> <api_secret> - Add user for auto-trading
/status - Show bot status
/stop_trading - Emergency stop all trading
""", parse_mode='Markdown')
    
    async def add_channel_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Add channel to monitoring"""
        if len(context.args) != 1:
            await update.message.reply_text("Usage: /add_channel <channel_username>")
            return
        
        channel_username = context.args[0]
        success = await self.channel_monitor.add_channel(channel_username)
        
        if success:
            await update.message.reply_text(f"✅ Channel {channel_username} added successfully!")
        else:
            await update.message.reply_text(f"❌ Failed to add channel {channel_username}")
    
    async def add_user_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Add user for auto-trading"""
        if len(context.args) != 3:
            await update.message.reply_text("Usage: /add_user <user_id> <api_key> <api_secret>")
            return
        
        user_id, api_key, api_secret = context.args
        
        try:
            # Test API credentials
            trader = EnhancedBybitTrader(api_key, api_secret, testnet=True)
            balance = await trader.get_account_balance()
            
            if balance:
                self.config_manager.add_user(int(user_id), api_key, api_secret)
                self.config_manager.update_user_setting(int(user_id), 'auto_trade', True)
                await update.message.reply_text(f"✅ User {user_id} added successfully for auto-trading!")
            else:
                await update.message.reply_text("❌ Invalid API credentials!")
                
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show bot status"""
        status_message = f"""
📊 **Bot Status**

🔍 **Monitored Channels:** {len(self.channel_monitor.monitored_channels)}
👥 **Active Users:** {len([u for u in self.config_manager.config.get('users', {}).values() if u.get('auto_trade')])}
🤖 **AI Analyzer:** ✅ Active
⚡ **Auto Trading:** ✅ Enabled

📈 **Recent Activity:**
- Signals processed today: {len([t for t in getattr(self, 'processed_signals', []) if t.timestamp.date() == datetime.now().date()])}
- Trades executed today: {len([t for trader in self.active_traders.values() for t in trader.trade_history if t['timestamp'].date() == datetime.now().date()])}
        """
        
        await update.message.reply_text(status_message, parse_mode='Markdown')
    
    async def stop_trading_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Emergency stop all trading"""
        try:
            # Disable auto trading for all users
            for user_id in self.config_manager.config.get('users', {}):
                self.config_manager.update_user_setting(int(user_id), 'auto_trade', False)
            
            await update.message.reply_text("🚨 **EMERGENCY STOP ACTIVATED**\n\nAll auto-trading has been disabled!", parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error stopping trading: {str(e)}")
    
    async def run(self):
        """Start the bot"""
        try:
            logger.info("Initializing Auto-Trading Bot...")
            
            # Initialize channel monitor
            if not await self.channel_monitor.initialize():
                raise Exception("Failed to initialize channel monitor")
            
            # Start admin bot in background
            admin_task = asyncio.create_task(self.admin_app.initialize())
            
            # Start channel monitoring
            monitor_task = asyncio.create_task(self.channel_monitor.start_monitoring())
            
            logger.info("🚀 Auto-Trading Bot started successfully!")
            logger.info("📡 Monitoring channels for AI-detected signals...")
            logger.info("🤖 Ready for automatic trade execution!")
            
            # Run both tasks concurrently
            await asyncio.gather(admin_task, monitor_task)
            
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            raise


# Enhanced Configuration Manager
class ConfigManager:
    """Enhanced configuration manager with auto-trading settings"""
    
    def __init__(self, config_file='config.yaml'):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load configuration from file"""
        default_config = {
            'users': {},
            'channels': {},
            'ai_settings': {
                'confidence_threshold': 70,
                'max_signals_per_hour': 10,
                'enable_risk_scaling': True
            },
            'risk_management': {
                'default_leverage': 10,
                'default_risk': 3.0,  # More conservative for auto-trading
                'max_leverage': 25,   # Lower max for safety
                'max_risk': 5.0,      # Lower max for safety
                'max_daily_loss': 100,  # USD
                'max_concurrent_trades': 3
            },
            'trading_hours': {
                'enabled': False,  # 24/7 by default for crypto
                'start_hour': 0,
                'end_hour': 23
            }
        }
        
        try:
            with open(self.config_file, 'r') as f:
                loaded_config = yaml.safe_load(f) or {}
                # Merge with defaults
                default_config.update(loaded_config)
                return default_config
        except FileNotFoundError:
            return default_config
    
    def save_config(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def add_user(self, user_id: int, api_key: str, api_secret: str):
        """Add user for auto-trading"""
        self.config['users'][str(user_id)] = {
            'api_key': api_key,
            'api_secret': api_secret,
            'leverage': self.config['risk_management']['default_leverage'],
            'risk': self.config['risk_management']['default_risk'],
            'auto_trade': False,  # Starts disabled for safety
            'testnet': True,      # Start with testnet
            'max_daily_loss': self.config['risk_management']['max_daily_loss'],
            'max_concurrent_trades': self.config['risk_management']['max_concurrent_trades'],
            'enabled_symbols': ['BTCUSDT', 'ETHUSDT'],  # Start with major pairs
            'created_at': datetime.now().isoformat()
        }
        self.save_config()
    
    def add_channel(self, channel_username: str, channel_title: str = ""):
        """Add channel to configuration"""
        self.config['channels'][channel_username] = {
            'title': channel_title,
            'added_at': datetime.now().isoformat(),
            'enabled': True,
            'signal_count': 0,
            'success_rate': 0.0
        }
        self.save_config()
    
    def get_user_config(self, user_id: int) -> Optional[Dict]:
        """Get user configuration"""
        return self.config['users'].get(str(user_id))
    
    def update_user_setting(self, user_id: int, setting: str, value):
        """Update user setting"""
        user_config = self.get_user_config(user_id)
        if user_config:
            user_config[setting] = value
            self.save_config()
            return True
        return False
    
    def get_ai_settings(self) -> Dict:
        """Get AI analyzer settings"""
        return self.config.get('ai_settings', {})
    
    def update_channel_stats(self, channel_username: str, signal_detected: bool, success: bool = None):
        """Update channel statistics"""
        if channel_username in self.config['channels']:
            channel = self.config['channels'][channel_username]
            if signal_detected:
                channel['signal_count'] += 1
            if success is not None:
                # Update success rate calculation
                pass
            self.save_config()


# Main execution function
async def main():
    """Main function to run the auto-trading bot"""
    
    # Load environment variables
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    TELEGRAM_API_ID = int(os.getenv('TELEGRAM_API_ID', '0'))
    TELEGRAM_API_HASH = os.getenv('TELEGRAM_API_HASH')
    PHONE_NUMBER = os.getenv('PHONE_NUMBER')
    
    # Validate required environment variables
    required_vars = {
        'TELEGRAM_BOT_TOKEN': TELEGRAM_BOT_TOKEN,
        'OPENAI_API_KEY': OPENAI_API_KEY,
        'TELEGRAM_API_ID': TELEGRAM_API_ID,
        'TELEGRAM_API_HASH': TELEGRAM_API_HASH,
        'PHONE_NUMBER': PHONE_NUMBER
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("\n📋 Required Environment Variables:")
        print("TELEGRAM_BOT_TOKEN=your_bot_token")
        print("OPENAI_API_KEY=your_openai_key")
        print("TELEGRAM_API_ID=your_api_id")
        print("TELEGRAM_API_HASH=your_api_hash")
        print("PHONE_NUMBER=your_phone_number")
        return
    
    try:
        # Create and run bot
        bot = AutoTradingBot(
            telegram_token=TELEGRAM_BOT_TOKEN,
            openai_api_key=OPENAI_API_KEY,
            telegram_api_id=TELEGRAM_API_ID,
            telegram_api_hash=TELEGRAM_API_HASH,
            phone_number=PHONE_NUMBER
        )
        
        await bot.run()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        raise


if __name__ == "__main__":
    # Run the bot
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        print("\n🔧 Check your configuration and try again")
