import re
import asyncio
import traceback
import ccxt.async_support as ccxt_async
import pandas as pd
import talib as ta
from datetime import datetime, timedelta, timezone
from app.core.logging import AppLogger

MIN_PRICE_CHANGE = 2

# Initialize logging
logger = AppLogger.get_logger()

# if sys.platform == 'win32':
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class BaseVolumeAnalyzer:
    def __init__(self, atr_period=14):
        self.atr_period = atr_period
        self.since = int((datetime.now(timezone.utc) - timedelta(days=1)).timestamp() * 1000)
        self.exchange = None
        self.df = pd.DataFrame()

    async def initialize(self):
        """Initialize the Binance futures exchange"""
        self.exchange = ccxt_async.binance({
            'options': {'defaultType': 'future'}
        })

    async def close(self):
        """Properly close the exchange connection"""
        if self.exchange:
            try:
                await self.exchange.close()
            except Exception as e:
                raise RuntimeError(f"Error closing exchange: {e}")
            finally:
                self.exchange = None

    async def get_futures_tickers(self):
        """Fetch all futures tickers"""
        if not self.exchange:
            raise RuntimeError("Exchange not initialized. Call initialize() first.")
        return await self.exchange.fetch_tickers()

    async def get_historical_data(self, symbol):
        """Fetch historical OHLCV data"""
        if not self.exchange:
            raise RuntimeError("Exchange not initialized. Call initialize() first.")

        ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe='15m', since=self.since)
        if ohlcv:
            self.df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='ms')
        else:
            self.df = pd.DataFrame()

    def calculate_atr(self, period=14):
        self.df['atr'] = ta.ATR(self.df['high'], self.df['low'], self.df['close'], timeperiod=period)
        self.df['atr_pct'] = (self.df['atr'] / self.df['close']) * 100

    def get_df(self):
        """Getter method to retrieve the DataFrame"""
        if self.df.empty:
            raise ValueError("DataFrame is empty. Please call get_historical_data first.")
        return self.df

    def calculate_support_resistance(self):
        """Calculate support and resistance levels"""
        self.df['pivot_point'] = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        self.df['r1'] = (2 * self.df['pivot_point']) - self.df['low']
        self.df['s1'] = (2 * self.df['pivot_point']) - self.df['high']
        self.df['r2'] = self.df['pivot_point'] + (self.df['high'] - self.df['low'])
        self.df['s2'] = self.df['pivot_point'] - (self.df['high'] - self.df['low'])
        self.df['r3'] = self.df['high'] + 2 * (self.df['pivot_point'] - self.df['low'])
        self.df['s3'] = self.df['low'] - 2 * (self.df['high'] - self.df['pivot_point'])


class BinanceVolumeAnalyzer(BaseVolumeAnalyzer):

    def __init__(self, atr_period=14):
        super().__init__(atr_period)
        self.df_final_values = pd.DataFrame()

    async def process_symbol(self, symbol):
        """Process individual symbol data"""
        await self.get_historical_data(symbol)
        if self.df is not None and len(self.df) > 1:
            self.calculate_atr(self.atr_period)
            df = self.df.copy()
            match = re.match(r"^[^/ \s]*", symbol)
            df['symbol'] = match.group(0) if match else symbol
            df['price_change'] = ta.ROC(df['close'].values, timeperiod=1)
            df['volume_change'] = ta.ROC(df['volume'].values, timeperiod=1)
            return df.iloc[[-1]]
        return pd.DataFrame()

    async def calculate_market_spikes(self):
        """Calculate volume changes and return top 3 results"""
        if not self.exchange:
            raise RuntimeError("Exchange not initialized. Call initialize() first.")

        dataframes = []
        tickers = await self.get_futures_tickers()

        # Filter USDT pairs
        usdt_pairs = [symbol for symbol in tickers.keys() if symbol.endswith('USDT')]

        tasks = [self.process_symbol(symbol) for symbol in usdt_pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for symbol, result in zip(usdt_pairs, results):
            if isinstance(result, Exception):
                logger.warning(f"[{symbol}] Task failed: {type(result).__name__} - {result}")
                logger.debug(traceback.format_exc())
                continue
            if isinstance(result, pd.DataFrame) and not result.empty:
                dataframes.append(result)

        if not dataframes:
            logger.info("No significant volume changes detected.")
            raise RuntimeWarning("No significant volume changes at the moment")

        # Concatenate and process results
        self.df_final_values = pd.concat(dataframes, ignore_index=True)

# TODO Please include the option to get a variable number of best and worst coins.
    def get_best_symbols(self, metrics="volume_change"):
        """Return the top 3 symbols based on the given metric"""
        if self.df_final_values.empty:
            return []

        df_best_symbols = (
            self.df_final_values
            .sort_values(by=metrics, ascending=False)
            .head(3)
            .reset_index(drop=True)
        )

        return df_best_symbols[
            ['symbol','price_change','volume_change','atr_pct', 'close', ]
        ].to_dict(orient='records')


def format_message_spikes(*args):
    """
    Formats message data from multiple dictionaries, filtering out messages
    where both price change is less than 3.5% and volume change is less than 1000.

    Args:
        *args: Variable number of dictionaries containing message data.
               Each dictionary should have the following keys:
               - 'symbol' (str): The symbol of the asset (e.g., "BTCUSD").
               - 'price_change' (float or str): The percentage price change.
               - 'volume_change' (float or str): The percentage volume change.
               - 'atr_pct' (float or str): The percentage of volatility using Average True Range.
               - 'close' (float or str): The closing price.

    Returns:
        str: A formatted string containing all the messages that meet the
             filtering criteria, separated by lines. If both price change
             is less than 3.5% and volume change is less than 1000,
             the message is skipped.
    """
    messages = []
    for data in args:
        # TODO: Implement a dynamic threshold as for now the values are fixed to 3.5 and 1000
        if all([float(data.get('price_change', '0')) < 3.5, float(data.get('volume_change', '0')) < 5000]):
            continue
        message = f"\nSymbol: {data.get('symbol', 'N/A')}\n"
        message += f"Price Change: {float(data.get('price_change', 0)):.2f}%\n"
        message += f"Volume Change: {float(data.get('volume_change', 0)):.2f}%\n"
        message += f"ATR Percentage: {float(data.get('atr_pct', '0')):.2f}%\n"
        message += f"Close Price: {data.get('close', 0)}\n"
        messages.append(message)

    return "\n--------\n".join(messages) if messages else ""

def format_symbol_name(symbol: str) -> str :
    if re.match(r"^[^/\s\d]*", symbol):
        return f'{symbol}/USDT:USDT'
    return ''
