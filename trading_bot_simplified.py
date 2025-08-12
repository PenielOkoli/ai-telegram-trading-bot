import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, List
import json
from dotenv import load_dotenv
import os

from telegram.ext import Application, CommandHandler
from pybit.unified_trading import HTTP
import openai
from telethon import TelegramClient, events
from telethon.tl.types import Channel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dataclasses import field

@dataclass
class TradingSignal:
    direction: str
    symbol: str
    confidence_score: float
    entry_price: Optional[float] = None
    take_profit: List[float] = None
    stop_loss: float = 0
    leverage: int = 10
    original_message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

class BybitTrader:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.session = HTTP(api_key=api_key, api_secret=api_secret, testnet=testnet)
        self.max_position_size = 10000.0  # Maximum position size in USDT
        self.max_daily_loss = 5.0  # Maximum daily loss percentage
        self.active_positions = {}
        self.daily_pnl = 0.0
        
    async def execute_trade(self, signal: TradingSignal, risk_pct: float = 1.0) -> Dict:
        try:
            # Validate signal first
            if not await self._validate_signal(signal):
                return {'success': False, 'error': 'Signal validation failed'}
            
            # Get balance and market data
            balance = self.session.get_wallet_balance(accountType="UNIFIED")
            market_data = await self._get_market_data(signal.symbol)
            
            if not market_data:
                return {'success': False, 'error': 'Could not fetch market data'}
                
            usdt_balance = next((float(coin['walletBalance']) for coin in balance['result']['list'][0]['coin'] 
                               if coin['coin'] == 'USDT'), 0)
            
            if usdt_balance < 10:
                return {'success': False, 'error': 'Insufficient balance'}
            
            # Calculate position size with volatility adjustment
            quantity = await self._calculate_position_size(signal, market_data, usdt_balance, risk_pct)
            
            # Check if position size exceeds limits
            position_value = quantity * market_data['price']
            if position_value > self.max_position_size:
                return {'success': False, 'error': 'Position size exceeds maximum limit'}
            
            # Place main order
            side = "Buy" if signal.direction in ["LONG", "BUY"] else "Sell"
            order_price = market_data['ask'] * 1.001 if side == "Buy" else market_data['bid'] * 0.999
            
            result = self.session.place_order(
                category="linear",
                symbol=signal.symbol,
                side=side,
                orderType="Limit",
                price=str(order_price),
                qty=str(round(quantity, 4)),
                timeInForce="PostOnly"
            )
            
            if result['retCode'] != 0:
                return {'success': False, 'error': result.get('retMsg')}
            
            order_id = result['result']['orderId']
            
            # Place take profit and stop loss orders
            if signal.take_profit:
                for tp in signal.take_profit:
                    self.session.place_order(
                        category="linear",
                        symbol=signal.symbol,
                        side="Sell" if side == "Buy" else "Buy",
                        orderType="Limit",
                        price=str(tp),
                        qty=str(round(quantity / len(signal.take_profit), 4)),
                        timeInForce="GoodTillCancel",
                        reduceOnly=True
                    )
            
            if signal.stop_loss:
                self.session.place_order(
                    category="linear",
                    symbol=signal.symbol,
                    side="Sell" if side == "Buy" else "Buy",
                    orderType="Stop",
                    stopPrice=str(signal.stop_loss),
                    basePrice=str(order_price),
                    qty=str(round(quantity, 4)),
                    timeInForce="GoodTillCancel",
                    reduceOnly=True,
                    triggerBy="LastPrice"
                )
            
            # Track position
            self.active_positions[signal.symbol] = {
                'order_id': order_id,
                'entry_price': order_price,
                'quantity': quantity,
                'side': side,
                'timestamp': datetime.now(),
                'take_profits': signal.take_profit,
                'stop_loss': signal.stop_loss,
                'leverage': signal.leverage,
                'realized_pnl': 0.0,
                'unrealized_pnl': 0.0
            }
            
            return {
                'success': True,
                'order_id': order_id,
                'quantity': quantity
            }
                
        except Exception as e:
            logger.error(f"Trade error: {e}")
            return {'success': False, 'error': str(e)}
            
    async def _validate_signal(self, signal: TradingSignal) -> bool:
        # Check if we already have a position
        if signal.symbol in self.active_positions:
            return False
            
        # Validate take profit and stop loss
        if not signal.take_profit or not signal.stop_loss:
            return False
            
        # Check if symbol has enough liquidity
        market_data = await self._get_market_data(signal.symbol)
        if not market_data or market_data['volume'] < 1000000:  # Minimum 24h volume in USDT
            return False
            
        return True
        
    async def _get_market_data(self, symbol: str) -> Optional[Dict]:
        try:
            ticker = self.session.get_tickers(category="linear", symbol=symbol)
            orderbook = self.session.get_orderbook(category="linear", symbol=symbol)
            
            return {
                'price': float(ticker['result']['list'][0]['lastPrice']),
                'bid': float(orderbook['result']['b'][0][0]),
                'ask': float(orderbook['result']['a'][0][0]),
                'volume': float(ticker['result']['list'][0]['volume24h'])
            }
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None
            
    async def _calculate_position_size(self, signal: TradingSignal, market_data: Dict, 
                                     balance: float, risk_pct: float) -> float:
        # Calculate stop loss distance
        if not signal.entry_price or not signal.stop_loss:
            return 0
            
        sl_distance = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
        if sl_distance == 0:
            return 0
            
        # Calculate risk amount in USDT
        risk_amount = balance * (risk_pct / 100)
        
        # Calculate position size based on stop loss
        position_size = risk_amount / (sl_distance * market_data['price'])
        
        # Apply leverage
        position_size *= signal.leverage
        
        # Adjust based on volatility
        price_change = abs(market_data['price'] - market_data['bid']) / market_data['price']
        if price_change > 0.001:  # More than 0.1% spread
            position_size *= 0.5  # Reduce position size for volatile markets
            
        return position_size
        
    async def update_position_pnl(self) -> None:
        """Update PnL for all active positions"""
        for symbol, position in list(self.active_positions.items()):
            try:
                market_data = await self._get_market_data(symbol)
                if not market_data:
                    continue
                
                current_price = market_data['price']
                price_diff = current_price - position['entry_price']
                if position['side'] == "Sell":
                    price_diff = -price_diff
                    
                # Calculate unrealized PnL
                position['unrealized_pnl'] = price_diff * position['quantity']
                
                # Check if stop loss or take profit hit
                if position['side'] == "Buy":
                    if current_price <= position['stop_loss'] or \
                       current_price >= min(position['take_profits']):
                        position['realized_pnl'] = position['unrealized_pnl']
                        del self.active_positions[symbol]
                else:  # Sell position
                    if current_price >= position['stop_loss'] or \
                       current_price <= max(position['take_profits']):
                        position['realized_pnl'] = position['unrealized_pnl']
                        del self.active_positions[symbol]
                        
                # Update daily PnL
                self.daily_pnl = sum(p['realized_pnl'] + p['unrealized_pnl'] 
                                   for p in self.active_positions.values())
                
            except Exception as e:
                logger.error(f"Error updating PnL for {symbol}: {e}")
                continue

class TradingBot:
    def __init__(self):
        load_dotenv()
        # Initialize with environment variables
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.api_id = int(os.getenv('TELEGRAM_API_ID', '0'))
        self.api_hash = os.getenv('TELEGRAM_API_HASH')
        self.phone = os.getenv('PHONE_NUMBER')
        
        # Setup components
        openai.api_key = self.openai_key
        
        # Initialize Telegram client with proper session handling
        session_file = os.path.join(os.path.dirname(__file__), 'trading_bot.session')
        self.client = TelegramClient(session_file, self.api_id, self.api_hash)
        
        # Initialize admin bot
        self.admin_app = Application.builder().token(self.bot_token).build()
        
        logger.info("Components initialized successfully")
        
        # Trading configuration
        self.users = {}  # {user_id: {'api_key': str, 'api_secret': str, 'enabled': bool}}
        self.monitored_channels = []
        self.symbol_whitelist = {'BTCUSDT', 'ETHUSDT'}
        self.symbol_blacklist = set()
        
        # Risk management
        self.daily_loss_limit = -5.0  # 5% max daily loss
        self.weekly_loss_limit = -15.0  # 15% max weekly loss
        self.max_positions = 3  # Maximum simultaneous positions
        self.trade_history = {}  # Track trades and performance
        
        # Setup command handlers
        commands = [
            ('start', self.cmd_start),
            ('add_user', self.cmd_add_user),
            ('add_channel', self.cmd_add_channel),
            ('status', self.cmd_status),
            ('list_channels', self.cmd_list_channels),
            ('check_connection', self.cmd_check_connection)
        ]
        for cmd_name, cmd_handler in commands:
            self.admin_app.add_handler(CommandHandler(cmd_name, cmd_handler))
            
        logger.info("Command handlers registered")

    async def analyze_message(self, message: str) -> Optional[TradingSignal]:
        if not any(kw in message.upper() for kw in ['LONG', 'SHORT', 'BUY', 'SELL']):
            return None
        
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[{
                    "role": "user", 
                    "content": f'Extract trading signal from: "{message}"\nReturn JSON only: {{"direction": "LONG/SHORT", "symbol": "BTCUSDT", "entry_price": null, "take_profit": [], "stop_loss": 0}}'
                }],
                max_tokens=100
            )
            
            data = json.loads(response.choices[0].message.content.strip())
            return TradingSignal(
                direction=data['direction'],
                symbol=data['symbol'].upper(),
                confidence_score=100,
                entry_price=data.get('entry_price'),
                take_profit=data.get('take_profit', []),
                stop_loss=data.get('stop_loss', 0)
            )
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return None

    async def handle_signal(self, signal: TradingSignal):
        # Validate signal
        if not await self._validate_trading_signal(signal):
            logger.info(f"Signal validation failed for {signal.symbol}")
            return
        
        # Check risk limits
        if not self._check_risk_limits():
            logger.info("Risk limits exceeded")
            return
        
        # Execute trades for each user
        for user_id, config in self.users.items():
            if not config.get('enabled'):
                continue
            
            try:
                trader = BybitTrader(config['api_key'], config['api_secret'])
                
                # Check user's existing positions
                if len(trader.active_positions) >= self.max_positions:
                    logger.info(f"Maximum positions reached for user {user_id}")
                    continue
                
                # Execute trade with volatility-adjusted risk
                result = await trader.execute_trade(signal)
                
                # Record trade
                if result['success']:
                    self._record_trade(user_id, signal, result)
                
                # Send notification
                msg = self._create_trade_notification(signal, result)
                if self.users:
                    await self.admin_app.bot.send_message(
                        chat_id=int(list(self.users.keys())[0]), 
                        text=msg
                    )
            except Exception as e:
                logger.error(f"Error for user {user_id}: {e}")
                
    async def _validate_trading_signal(self, signal: TradingSignal) -> bool:
        # Check whitelist/blacklist
        if (self.symbol_whitelist and signal.symbol not in self.symbol_whitelist) or \
           signal.symbol in self.symbol_blacklist:
            return False
        
        # Validate take profit and stop loss
        if not signal.take_profit or not signal.stop_loss or not signal.entry_price:
            return False
            
        # Validate TP/SL levels
        tp_valid = all(abs(tp - signal.entry_price) / signal.entry_price <= 0.1 
                      for tp in signal.take_profit)
        sl_valid = abs(signal.stop_loss - signal.entry_price) / signal.entry_price <= 0.1
        
        return tp_valid and sl_valid
    
    def _check_risk_limits(self) -> bool:
        daily_pnl = sum(trade['pnl'] for trades in self.trade_history.values() 
                       for trade in trades if trade['timestamp'].date() == datetime.now().date())
        
        weekly_pnl = sum(trade['pnl'] for trades in self.trade_history.values() 
                        for trade in trades 
                        if (datetime.now() - trade['timestamp']).days <= 7)
        
        return daily_pnl > self.daily_loss_limit and weekly_pnl > self.weekly_loss_limit
    
    def _record_trade(self, user_id: str, signal: TradingSignal, result: Dict):
        if user_id not in self.trade_history:
            self.trade_history[user_id] = []
            
        self.trade_history[user_id].append({
            'timestamp': datetime.now(),
            'symbol': signal.symbol,
            'direction': signal.direction,
            'quantity': result['quantity'],
            'order_id': result['order_id'],
            'entry_price': signal.entry_price,
            'take_profit': signal.take_profit,
            'stop_loss': signal.stop_loss,
            'pnl': 0.0  # To be updated when position is closed
        })
    
    def _create_trade_notification(self, signal: TradingSignal, result: Dict) -> str:
        status = "✅" if result['success'] else "❌"
        msg = f"{status} {signal.symbol} {signal.direction}\n"
        msg += f"Entry: {signal.entry_price}\n"
        if signal.take_profit:
            msg += f"TP: {', '.join(map(str, signal.take_profit))}\n"
        if signal.stop_loss:
            msg += f"SL: {signal.stop_loss}"
        return msg

    async def cmd_start(self, update, context):
        await update.message.reply_text("Bot ready! Use /add_user and /add_channel to configure.")

    async def cmd_add_user(self, update, context):
        if len(context.args) != 3:
            await update.message.reply_text("Usage: /add_user <user_id> <api_key> <api_secret>")
            return
        
        user_id, api_key, api_secret = context.args
        self.users[user_id] = {'api_key': api_key, 'api_secret': api_secret, 'enabled': True}
        await update.message.reply_text(f"User {user_id} added!")

    async def cmd_add_channel(self, update, context):
        if not context.args:
            await update.message.reply_text("Usage: /add_channel <channel_username>")
            return
        
        try:
            channel_username = context.args[0]
            logger.info(f"Attempting to add channel: {channel_username}")
            
            entity = await self.client.get_entity(channel_username)
            if isinstance(entity, Channel):
                if entity not in self.monitored_channels:
                    self.monitored_channels.append(entity)
                    await update.message.reply_text(
                        f"✅ Channel {channel_username} added successfully!\n"
                        f"Use /list_channels to see all monitored channels"
                    )
                else:
                    await update.message.reply_text("This channel is already being monitored!")
            else:
                await update.message.reply_text("❌ Error: This is not a valid channel")
        except ValueError as e:
            await update.message.reply_text(f"❌ Invalid channel username: {str(e)}")
        except Exception as e:
            logger.error(f"Error adding channel {context.args[0]}: {str(e)}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
            
    async def cmd_list_channels(self, update, context):
        if not self.monitored_channels:
            await update.message.reply_text("No channels are currently being monitored.")
            return
            
        channels_list = "\n".join(f"- {channel.username or channel.title}" 
                                for channel in self.monitored_channels)
        await update.message.reply_text(
            f"📊 Monitored Channels:\n{channels_list}"
        )
        
    async def cmd_check_connection(self, update, context):
        status = []
        
        # Check Telegram client
        try:
            if await self.client.is_user_authorized():
                me = await self.client.get_me()
                status.append(f"✅ Telegram Client: Connected as {me.first_name}")
            else:
                status.append("❌ Telegram Client: Not authorized")
        except Exception as e:
            status.append(f"❌ Telegram Client Error: {str(e)}")
            
        # Check OpenAI
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            status.append("✅ OpenAI API: Connected")
        except Exception as e:
            status.append(f"❌ OpenAI API Error: {str(e)}")
            
        # Report status
        await update.message.reply_text("\n".join(status))

    async def run(self):
        try:
            # Validate environment variables
            if not all([self.bot_token, self.openai_key, self.api_id, self.api_hash, self.phone]):
                logger.error("Missing environment variables! Please check your .env file")
                return

            # Initialize Telegram client with error handling
            try:
                logger.info("Starting Telegram client...")
                await self.client.connect()
                
                if not await self.client.is_user_authorized():
                    logger.info("First time login, sending code request...")
                    await self.client.send_code_request(self.phone)
                    logger.info("Please check your Telegram app for the code and run the bot again")
                    return
                
                logger.info("Successfully connected to Telegram!")
            except Exception as e:
                logger.error(f"Failed to initialize Telegram client: {str(e)}")
                return

            # Set up message handler
            @self.client.on(events.NewMessage())
            async def handle_message(event):
                try:
                    if any(event.chat_id == channel.id for channel in self.monitored_channels):
                        logger.info(f"Received message from monitored channel: {event.message.text[:100]}...")
                        signal = await self.analyze_message(event.message.message)
                        if signal:
                            logger.info(f"Valid trading signal detected for {signal.symbol}")
                            await self.handle_signal(signal)
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")

            # Initialize admin bot
            try:
                logger.info("Starting admin bot...")
                admin_task = asyncio.create_task(self.admin_app.initialize())
                logger.info("Admin bot started successfully!")
            except Exception as e:
                logger.error(f"Failed to start admin bot: {str(e)}")
                return

            logger.info("🚀 Trading bot is ready and monitoring channels!")
            
            # Run both clients
            await asyncio.gather(
                admin_task,
                self.client.run_until_disconnected()
            )
            
        except Exception as e:
            logger.error(f"Critical error in bot execution: {str(e)}")
            raise

if __name__ == "__main__":
    bot = TradingBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("Bot stopped!")