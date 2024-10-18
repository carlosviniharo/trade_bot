import sys
import asyncio
import ccxt.async_support as ccxt_async
import pandas as pd
import talib as ta
from datetime import datetime, timedelta, timezone

ATR_PERIOD = 14

# Add this code block right after the imports
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Initialize Binance futures client
exchange = ccxt_async.binance({
    'options': {'defaultType': 'future'}
})


async def get_futures_tickers():
    tickers = await exchange.fetch_tickers()
    return tickers


async def get_historical_data(symbol, since):
    # Fetch historical OHLCV data (15-minute intervals)
    ohlcv = await exchange.fetch_ohlcv(symbol, timeframe='15m', since=since)
    if ohlcv:
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    return None


async def process_symbol(symbol, since):
    df = await get_historical_data(symbol, since)
    if df is not None and len(df) > 1:
        df['symbol'] = symbol
        df['price_change'] = ta.ROC(df['close'].values, timeperiod=1)
        df['volume_change'] = ta.ROC(df['volume'].values, timeperiod=1).round(2)
        df['ATR'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=ATR_PERIOD)
        df['precentageATR'] = (df['ATR'] / df['close']) * 100

        if abs(df['price_change'].iloc[-1]) >= max(df['precentageATR'].iloc[-1], 1.5):
            df['ema_9'] = ta.EMA(df['close'].values, timeperiod=9)
            return df.iloc[[-1]]
    return None


def calculate_support_resistance(df):
    # Calculate support and resistance levels using pivot points
    df['pivot_point'] = (df['high'] + df['low'] + df['close']) / 3
    df['r1'] = (2 * df['pivot_point']) - df['low']
    df['s1'] = (2 * df['pivot_point']) - df['high']
    df['r2'] = df['pivot_point'] + (df['high'] - df['low'])
    df['s2'] = df['pivot_point'] - (df['high'] - df['low'])
    df['r3'] = df['high'] + 2 * (df['pivot_point'] - df['low'])
    df['s3'] = df['low'] - 2 * (df['high'] - df['pivot_point'])

    return df


async def calculate_volume_changes():
    tickers = await get_futures_tickers()
    df_final_values = pd.DataFrame()

    now = datetime.now(timezone.utc)
    since = int((now - timedelta(days=1)).timestamp() * 1000)

    # Filter USDT pairs before the loop
    usdt_pairs = [symbol for symbol in tickers.keys() if symbol.endswith('USDT')]

    tasks = [process_symbol(symbol, since) for symbol in usdt_pairs]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, pd.DataFrame):
            df_final_values = pd.concat([df_final_values, result])

    if df_final_values.empty:
        return "No significant volume changes at the moment"

    df_final_values.sort_values(by='volume_change', ascending=False, inplace=True)
    df_final_values = df_final_values.head(3)  # Keep top 3

    df_final_values = calculate_support_resistance(df_final_values)
    df_final_values.reset_index(drop=True, inplace=True)

    # Log and return top 3 coins with the highest volume change as a dictionary
    result_dict = df_final_values[
        ['symbol', 'volume_change', 'close', 'r1', 's1', 'r2', 's2', 'r3', 's3']].to_dict(orient='records')

    return result_dict