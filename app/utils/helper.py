import asyncio
import re
import traceback
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Callable, Union, Tuple

import requests
import ccxt.async_support as ccxt_async
import pandas as pd
import talib as ta
from fastapi import Query

from app.core.logging import AppLogger
MIN_PRICE_CHANGE = 2

# Initialize logging
logger = AppLogger.get_logger()

# if sys.platform == 'win32':
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class BaseVolumeAnalyzer:
    """Base class for volume analysis with technical indicators."""
    
    def __init__(self, atr_period: int = 14) -> None:
        self.atr_period = atr_period
        self.since = int((datetime.now(timezone.utc) - timedelta(days=1)).timestamp() * 1000)
        self.exchange = None
        self.df = pd.DataFrame()

    async def initialize(self) -> None:
        """Initialize the Binance futures exchange."""
        self.exchange = ccxt_async.binance({
            'options': {'defaultType': 'future'}
        })

    async def close(self) -> None:
        """Properly close the exchange connection."""
        if self.exchange:
            try:
                await self.exchange.close()
            except Exception as e:
                raise RuntimeError(f"Error closing exchange: {e}")
            finally:
                self.exchange = None

    async def get_futures_tickers(self) -> Dict[str, Any]:
        """
        Fetch all future tickers.
        
        Raises:
            RuntimeError: If exchange is not initialized.
        """
        if not self.exchange:
            raise RuntimeError("Exchange not initialized. Call initialize() first.")
        return await self.exchange.fetch_tickers()

    async def get_historical_data(self, symbol: str) -> None:
        """
        Fetch historical OHLCV data.
        
        Raises:
            RuntimeError: If exchange is not initialized.
        """
        if not self.exchange:
            raise RuntimeError("Exchange not initialized. Call initialize() first.")

        ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe='15m', since=self.since)
        if ohlcv:
            self.df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            self.df['event_timestamp'] = pd.to_datetime(self.df['timestamp'], unit='ms')
        else:
            self.df = pd.DataFrame()

    def calculate_atr(self, period: int = 14) -> None:
        """Calculate Average True Range (ATR) indicator."""
        self.df['atr'] = ta.ATR(self.df['high'], self.df['low'], self.df['close'], timeperiod=period)
        self.df['atr_pct'] = (self.df['atr'] / self.df['close']) * 100

    def get_df(self) -> pd.DataFrame:
        """
        Getter method to retrieve the DataFrame.
        
        Raises:
            ValueError: If DataFrame is empty.
        """
        if self.df.empty:
            raise ValueError("DataFrame is empty. Please call get_historical_data first.")
        return self.df

    def calculate_support_resistance(self) -> None:
        """Calculate support and resistance levels."""
        self.df['pivot_point'] = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        self.df['r1'] = (2 * self.df['pivot_point']) - self.df['low']
        self.df['s1'] = (2 * self.df['pivot_point']) - self.df['high']
        self.df['r2'] = self.df['pivot_point'] + (self.df['high'] - self.df['low'])
        self.df['s2'] = self.df['pivot_point'] - (self.df['high'] - self.df['low'])
        self.df['r3'] = self.df['high'] + 2 * (self.df['pivot_point'] - self.df['low'])
        self.df['s3'] = self.df['low'] - 2 * (self.df['high'] - self.df['pivot_point'])


class BinanceVolumeAnalyzer(BaseVolumeAnalyzer):
    """Extended volume analyzer for Binance with market spike detection."""

    def __init__(self, atr_period: int = 14) -> None:
        super().__init__(atr_period)
        self.df_final_values = pd.DataFrame()

    async def process_symbol(self, symbol: str) -> pd.DataFrame:
        """Process individual symbol data."""
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

    async def calculate_market_spikes(self) -> None:
        """
        Process all USDT futures pairs, calculate price and volume changes,
        and aggregate the latest results. Raises if no significant changes found.
        """
        if not self.exchange:
            raise RuntimeError("Exchange not initialized. Call initialize() first.")

        dataframes = []
        tickers = await self.get_futures_tickers()

        # Select only symbols that are USDT pairs
        usdt_pairs = [symbol for symbol in tickers.keys() if symbol.endswith('USDT')]

        # Asynchronously process each USDT pair to compute indicators
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

        # Combine all processed DataFrames into one for further use
        self.df_final_values = pd.concat(dataframes, ignore_index=True)

# TODO Please include the option to get a variable number of best and worst coins.
    def get_top_symbols(
        self, 
        metric: str = "volume_change", 
        ascending: bool = True, 
        limit: int = 3
    ) -> list[dict[str, Any]]:
        """
        Return the top symbols based on the given metric.

        Improvements:
        - Handles missing/invalid metric gracefully.
        - Adds event flags per record, not to the whole result.
        - Avoids mutating the result structure in a way that breaks list-of-dict contract.
        - Ensures only valid columns are selected.
        - Returns an empty list if metric is not present.
        """
        if self.df_final_values.empty:
            return []

        if metric not in self.df_final_values.columns:
            logger.warning(f"Metric '{metric}' not found in DataFrame columns.")
            return []

        # Select only the relevant columns that exist in the DataFrame
        self.df_final_values = self.df_final_values[['symbol', 'event_timestamp', 'price_change', 'volume_change', 'atr_pct', 'close']]

        df_sorted = (
            self.df_final_values
            .sort_values(by=metric, ascending=ascending)
            .head(limit)
            .reset_index(drop=True)
        )

        if metric == "price_change":
            df_sorted["is_price_event"] = True
        elif metric == "volume_change":
            df_sorted["is_volume_event"] = True

        return df_sorted.to_dict(orient='records')


def format_message_spikes(*args: Dict[str, Any]) -> str:
    """
    Formats message data from multiple dictionaries, filtering out messages
    where both price changes are less than 3.5% and volume change is less than 1000.

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

        if abs(price_change) < 3 and volume_change < 5000:
            continue

        # Use Telegram-compatible HTML formatting (no <hr />, only supported tags)
        message = (
            f"\nSymbol: {raw.get('symbol', 'N/A')}\n"
            f"Price Change: {price_change:.2f}%\n"
            f"Volume Change: {volume_change:.2f}%\n"
            f"ATR Percentage: {atr_pct:.2f}%\n"
            f"Close Price: {close}\n"
            f"──────────────"
        )
        messages += message

    return messages

def format_symbol_name(symbol: str) -> str:
    """Format symbol name for trading."""
    if re.match(r"^[^/\s\d]*", symbol, re.IGNORECASE):
        return f'{symbol.upper()}/USDT:USDT'
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

    def __init__(self, now_provider: Callable[[], datetime] = datetime.now) -> None:
        self._now = now_provider

    def fetch_market_data(self) -> Optional[List[Dict[str, Any]]]:
        """Fetch market data from CoinGecko API."""
        try:
            response = requests.get(self._COINGECKO_API_URL, params=self._DEFAULT_PARAMS, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"[ERROR] Unable to fetch market data: {e}")
            return None

    def calculate_weighted_sentiment(self, market_data: List[Dict[str, Any]]) -> Optional[float]:
        """Calculate weighted sentiment based on market cap and price changes."""
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
        """Render sentiment report as formatted string."""
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

class PaginationParams:
    """Pagination parameters for API endpoints."""
    
    def __init__(
            self,
            page: int = Query(1, ge=1),
            limit: int = Query(10, ge=1, le=100)
    ) -> None:
        self.page = page
        self.limit = limit

    @property
    def skip(self) -> int:
        """Calculate skip value for pagination."""
        return (self.page - 1) * self.limit
    