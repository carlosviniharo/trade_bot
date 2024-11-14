import re
import asyncio
import sys

import ccxt.async_support as ccxt_async
import pandas as pd
import talib as ta
from datetime import datetime, timedelta, timezone


if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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
            await self.exchange.close()
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
        self.df['ATR'] = ta.ATR(self.df['high'], self.df['low'], self.df['close'], timeperiod=period)
        self.df['percentageATR'] = (self.df['ATR'] / self.df['close']) * 100

    def get_df(self):
        """Getter method to retrieve the DataFrame"""
        if self.df.empty:
            raise ValueError("DataFrame is empty. Please call get_historical_data first.")
        return self.df


    @staticmethod
    def calculate_support_resistance(df):
        """Calculate support and resistance levels"""
        df['pivot_point'] = (df['high'] + df['low'] + df['close']) / 3
        df['r1'] = (2 * df['pivot_point']) - df['low']
        df['s1'] = (2 * df['pivot_point']) - df['high']
        df['r2'] = df['pivot_point'] + (df['high'] - df['low'])
        df['s2'] = df['pivot_point'] - (df['high'] - df['low'])
        df['r3'] = df['high'] + 2 * (df['pivot_point'] - df['low'])
        df['s3'] = df['low'] - 2 * (df['high'] - df['pivot_point'])
        return df


class BinanceVolumeAnalyzer(BaseVolumeAnalyzer):

    def __init__(self, atr_period=14):
        super().__init__(atr_period)
        self.df_final_values = pd.DataFrame()


    async def process_symbol(self, symbol):
        """Process individual symbol data"""
        await self.get_historical_data(symbol)
        if self.df is not None and len(self.df) > 1:
            self.df['symbol'] = re.match(r"^[^/ \s]*", symbol).group(0)
            self.df['price_change'] = ta.ROC(self.df['close'].values, timeperiod=1)
            self.df['volume_change'] = ta.ROC(self.df['volume'].values, timeperiod=1).round(2)

            self.calculate_atr(self.atr_period)

            if abs(self.df['price_change'].iloc[-1]) > min(self.df['percentageATR'].iloc[-1], 2):
                self.df['ema_9'] = ta.EMA(self.df['close'].values, timeperiod=9)
                return self.df.iloc[[-1]]
        return None

    async def calculate_volume_changes(self):
        """Calculate volume changes and return top 3 results"""
        if not self.exchange:
            raise RuntimeError("Exchange not initialized. Call initialize() first.")

        tickers = await self.get_futures_tickers()

        # Filter USDT pairs
        usdt_pairs = [symbol for symbol in tickers.keys() if symbol.endswith('USDT')]

        tasks = [self.process_symbol(symbol) for symbol in usdt_pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful DataFrame results
        dataframes = [result for result in results if isinstance(result, pd.DataFrame)]
        if not dataframes:
            return "No significant volume changes at the moment"

        # Concatenate and process results
        self.df_final_values = pd.concat(dataframes, ignore_index=True)
        self.df_final_values = (
           self. df_final_values
            .sort_values(by='volume_change', ascending=False)
            .head(3)
            .reset_index(drop=True)
        )

        self.df_final_values = self.calculate_support_resistance(self.df_final_values)

        return self.df_final_values[
            ['symbol', 'volume_change', 'close', 'r1', 's1', 'r2', 's2', 'r3', 's3']
        ].to_dict(orient='records')