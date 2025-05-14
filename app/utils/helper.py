import asyncio
import re
import traceback
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict

import requests
import ccxt.async_support as ccxt_async
import pandas as pd
import talib as ta

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
             filtering criteria, separated by lines.
    """
    seen = set()
    messages = ""
    for raw in args:
        key = frozenset(sorted(raw.items()))
        if key in seen:
            continue
        seen.add(key)

        try:
            price_change = float(raw.get('price_change', 0))
            volume_change = float(raw.get('volume_change', 0))
            atr_pct = float(raw.get('atr_pct', 0))
            close = float(raw.get('close', 0))
        except ValueError:
            continue

        if price_change < 3.5 and volume_change < 2000:
            continue

        message = (
            f"\nSymbol: {raw.get('symbol', 'N/A')}\n"
            f"Price Change: {price_change:.2f}%\n"
            f"Volume Change: {volume_change:.2f}%\n"
            f"ATR Percentage: {atr_pct:.2f}%\n"
            f"Close Price: {close}"
            f"--------\n"
        )
        messages += message

    return messages

def format_symbol_name(symbol: str) -> str :
    if re.match(r"^[^/\s\d]*", symbol):
        return f'{symbol}/USDT:USDT'
    return ''


class MarketSentimentAnalyzer:
    """
    MarketSentimentAnalyzer encapsulates logic to retrieve and analyze
    cryptocurrency market sentiment based on 24h price changes weighted by market cap.
    """

    _COINGECKO_API_URL = "https://api.coingecko.com/api/v3/coins/markets"
    _STABLECOINS = {"usdt", "usdc", "busd", "dai", "tusd", "usdp"}
    _DEFAULT_PARAMS = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 250,
        "page": 1
    }

    def __init__(self, now_provider=datetime.now):
        self._now = now_provider

    def fetch_market_data(self) -> Optional[List[Dict]]:
        try:
            response = requests.get(self._COINGECKO_API_URL, params=self._DEFAULT_PARAMS, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"[ERROR] Unable to fetch market data: {e}")
            return None

    def calculate_weighted_sentiment(self, market_data: List[Dict]) -> Optional[float]:
        total_market_cap = 0.0
        weighted_change_sum = 0.0

        for coin in market_data:
            symbol = coin.get("symbol", "").lower()
            market_cap = coin.get("market_cap")
            change_24h = coin.get("price_change_percentage_24h")

            if not symbol or market_cap is None or change_24h is None or symbol in self._STABLECOINS:
                continue

            total_market_cap += market_cap
            weighted_change_sum += market_cap * change_24h

        if total_market_cap == 0:
            return None

        return weighted_change_sum / total_market_cap

    def render_report(self, sentiment_score: Optional[float]) -> str:
        timestamp = self._now().isoformat(sep=' ', timespec='seconds')
        report = "\n--- Market Sentiment Report ---"
        report += f"\nTimestamp: {timestamp}"

        if sentiment_score is None:
            report += "\n[INFO] Insufficient data to determine market sentiment."
            return report

        trend_label = "🚀🌕 Bullish" if sentiment_score > 0 else "⚠️ Bearish"
        report += f"\nOverall Market Sentiment (24h): {sentiment_score:.2f}%"
        report += f"\nMarket Sentiment: {trend_label}"
        return report
