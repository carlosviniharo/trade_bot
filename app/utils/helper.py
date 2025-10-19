import asyncio
import re
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import talib as ta
from fastapi import Query
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from statsmodels.tsa.stattools import acf
from xgboost import XGBRegressor

import ccxt.async_support as ccxt_async
from app.core.logging import AppLogger

MIN_PRICE_CHANGE = 3
MIN_VOLUME_CHANGE = 5000 # Notice that the volumen movements are above 100 then 5000 is a good threshold.
## The time limit should be calcuated dimanically based on the timeframe.
## For example, if the timeframe is 15m, the limit should be 96.
## If the timeframe is 1h, the limit should be 24.
## If the timeframe is 4h, the limit should be 6.
## If the timeframe is 1d, the limit should be 1.
SINCE_24H_AGO_LIMIT = 96
# Initialize logging
logger = AppLogger.get_logger()

# if sys.platform == 'win32':
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class BaseAnalyzer:
    """Base class for volume analysis with technical indicators."""
    
    def __init__(
            self, 
            exchange_id: str = "binance", 
            timeframe: str = '15m', 
            limit: int = SINCE_24H_AGO_LIMIT,
            ) -> None:
        self.limit = limit
        self.exchange = None
        self.exchange_id = exchange_id
        self.timeframe = timeframe
        self.df = pd.DataFrame()


    async def initialize(self) -> None:
        """Initialize the Binance futures exchange."""

        if self.exchange_id == "binance":
            self.exchange = ccxt_async.binance({
                'options': {'defaultType': 'future'}
            })
        elif self.exchange_id == "bybit":
            self.exchange = ccxt_async.bybit({
                'options': {'defaultType': 'future'}
            })
        elif self.exchange_id == "okx":
            self.exchange = ccxt_async.okx({
                'options': {'defaultType': 'future'}
            })
        else:
            raise ValueError(f"Invalid exchange ID: {self.exchange_id}")

    async def close(self) -> None:
        """Properly close the exchange connection and underlying aiohttp session."""
        if self.exchange:
            try:
                # Try ccxt's own close (closes session in most cases)
                await self.exchange.close()
                
                # If ccxt left the session open, close it manually
                session = getattr(self.exchange, "session", None)
                if session and not session.closed:
                    await session.close()
                    logger.info("aiohttp session closed manually.")
                    
            except Exception as e:
                logger.error(f"Error during exchange close: {type(e).__name__} - {e}")
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

        ohlcv = await self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=self.limit)
        if ohlcv:
            self.df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            self.df['event_timestamp'] = pd.to_datetime(self.df['timestamp'], unit='ms')

        else:
            self.df = pd.DataFrame()

    def calculate_atr(self, window: int = 14) -> None:
        """Calculate Average True Range (ATR) indicator."""
        self.df['atr'] = ta.ATR(self.df['high'], self.df['low'], self.df['close'], timeperiod=window)
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

    def get_rsi(self, metric: str = "close", window: int = 14):
        """
        Get RSI indicator.
        """
        metric_values = self.df[metric].values
        self.df["rsi"] = ta.RSI(metric_values, timeperiod=window)
        
    def get_roc(self, metric: str = "close", window: int = 10):
        """
        Get ROC indicator.
        """
        metric_values = self.df[metric].values
        if metric == "close":
            self.df["price_rate"] = ta.ROC(metric_values, timeperiod=window)
        else:
            self.df[f"{metric}_rate"] = ta.ROC(metric_values, timeperiod=window)

    def get_macd(self, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9):
        """
        Get MACD indicator.
        """
        close = self.df["close"].values

        macd, macd_signal, macd_hist = ta.MACD(
            close, 
            fastperiod=fastperiod, 
            slowperiod=slowperiod, 
            signalperiod=signalperiod
            )
        self.df["macd"] = macd
        self.df["macd_signal"] = macd_signal
        self.df["macd_hist"] = macd_hist
    
    def get_atr(self, window: int = 14):
        """
        Get ATR indicator.
        """
        high = self.df["high"].values
        low = self.df["low"].values
        close = self.df["close"].values
        
        self.df['atr'] = ta.ATR(high, low, close, timeperiod=window)
        self.df['atr_pct'] = (self.df['atr'] / close) * 100

    def get_atr_above_median(self):
        if not "atr" in self.df:
            msg = "ATR not found in DataFrame. Please call get_atr first."
            logger.error(msg)
            raise ValueError(msg)
        median_window = calculate_correlation(self.df)

        if len(self.df) <= median_window:
            self.df['atr_mean'] = 0
            self.df['atr_above_mean'] = False
            logger.info("Not enough data to compute ATR median.")
        else:
            self.df['atr_mean'] = self.df['atr'].rolling(window=median_window).mean()
            self.df['atr_above_mean'] = self.df['atr'] > self.df['atr_mean']

    def get_bbands(self, window: int = 20):
        """
        Get Bollinger Bands indicator.
        """
        close = self.df["close"].values
        self.df["bb_high"], self.df["bb_mid"], self.df["bb_low"] = ta.BBANDS(close, timeperiod=window)
        self.df["bb_width"] = (self.df["bb_high"] - self.df["bb_low"]) / self.df["bb_mid"]

    def get_ema(self, window: int = 20):
        """
        Get EMA indicator.
        """
        close = self.df["close"].values
        self.df[f"ema{window}"] = ta.EMA(close, timeperiod=window)

    def get_obv(self):
        """
        Get OBV indicator.
        """
        close = self.df["close"].values
        volume = self.df["volume"].values
        self.df["obv"] = ta.OBV(close, volume)
    
    def get_vol_sma(self, window: int = 20):
        """
        Get Volume SMA indicator.
        """
        volume = self.df["volume"].values
        self.df["vol_sma"] = pd.Series(volume).rolling(window).mean().values


class BinanceVolumeAnalyzer(BaseAnalyzer):
    """Extended volume analyzer for Binance with market spike detection."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.df_final_values = pd.DataFrame()

    async def process_symbol(self, symbol: str, window: int = 1) -> pd.DataFrame:
        """Process individual symbol data."""
        await self.get_historical_data(symbol)
        if self.df is not None and len(self.df) > 1:
            self.get_atr()
            self.get_roc("close", window)
            self.get_roc("volume", window)
            result = self.df.copy()
            match = re.match(r"^[^/ \s]*", symbol)
            result['symbol'] = match.group(0) if match else symbol

            return result.iloc[[-1]]
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
        
        if not usdt_pairs:
            logger.warning("No USDT pairs found in tickers.")
            raise RuntimeWarning("No USDT pairs available for analysis")

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


    def get_top_symbols(
        self, 
        metric: str = "volume_rate", 
        ascending: bool = False,
        n_values: int = 3
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
            logger.warning("DataFrame is empty. Please call calculate_market_spikes first.")
            raise ValueError("DataFrame is empty. Please call calculate_market_spikes first.")

        if metric not in self.df_final_values.columns:
            logger.warning(f"Metric '{metric}' not found in DataFrame columns.")
            raise ValueError(f"Metric '{metric}' not found in DataFrame columns.")

        df_sorted = (
            self.df_final_values[['symbol', 'event_timestamp', 'price_rate', 'volume_rate', 'atr_pct', 'close']].copy()
            .sort_values(by=metric, ascending=ascending)
            .head(n_values)
            .reset_index(drop=True)
        )

        # Initialize both columns as False
        df_sorted["is_price_event"] = False
        df_sorted["is_volume_event"] = False

        if metric == "price_rate":
            df_sorted["is_price_event"] = True
        elif metric == "volume_rate":
            df_sorted["is_volume_event"] = True

        return df_sorted


class XGBoostSupportResistancePredictor(BaseAnalyzer):
    """
    XGBoostSupportResistancePredictor uses XGBoost machine learning to predict
    cryptocurrency support and resistance levels using technical indicators.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.df_final = pd.DataFrame()

    async def get_historical_data(self, symbol: str) -> None:
        ohlcv = await self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=self.limit)
        if ohlcv:
            self.df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            self.df['event_timestamp'] = pd.to_datetime(self.df['timestamp'], unit='ms')
            self.df.set_index("timestamp", inplace=True)
        else:
            self.df = pd.DataFrame()
        
# TODO: Include a exception indicating to run this method first in order to train the model.
    def add_technical_indicators(self):
        """
        Add technical indicators to the dataframe.
        """
        self.df_final = self.df.copy()
        close = self.df_final["close"].values
        high = self.df_final["high"].values
        low = self.df_final["low"].values
        volume = self.df_final["volume"].values

        # RSI calculation
        self.df_final["rsi"] = ta.RSI(close, timeperiod=14)

        # ROC calculation
        self.df_final["roc"] = ta.ROC(close, timeperiod=10)

        # MACD calculation
        macd, macd_signal, macd_hist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        self.df_final["macd"] = macd
        self.df_final["macd_signal"] = macd_signal
        self.df_final["macd_hist"] = macd_hist

        # ATR calculation
        self.df_final["atr"] = ta.ATR(high, low, close, timeperiod=14)

        # Bollinger Bands calculation
        self.df_final["bb_high"], self.df_final["bb_mid"], self.df_final["bb_low"] = ta.BBANDS(close, timeperiod=20)
        self.df_final["bb_width"] = (self.df_final["bb_high"] - self.df_final["bb_low"]) / self.df_final["bb_mid"]

        # EMA calculations
        self.df_final["ema20"] = ta.EMA(close, timeperiod=20)
        self.df_final["ema50"] = ta.EMA(close, timeperiod=50)

        # OBV calculation
        self.df_final["obv"] = ta.OBV(close, volume)

        # Volume SMA calculation
        self.df_final["vol_sma"] = pd.Series(volume).rolling(20).mean().values
        
        
        self.df_final.dropna(inplace=True)
    
    def generate_targets(self, window=10, quantile_level=0.8):

        self.df_final["future_max"] = (
            self.df_final["high"].shift(-1).rolling(window).quantile(quantile_level)
            )
        self.df_final["future_min"] = (
            self.df_final["low"].shift(-1).rolling(window).quantile(1 - quantile_level)
            )
        # Fill NaNs with current high/low to avoid dropping rows
        self.df_final["future_max"] = self.df_final["future_max"].fillna(self.df_final["high"])
        self.df_final["future_min"] = self.df_final["future_min"].fillna(self.df_final["low"])

        # Transform to log space (important!)
        self.df_final["future_max_log"] = np.log(self.df_final["future_max"])
        self.df_final["future_min_log"] = np.log(self.df_final["future_min"])
        self.df_final.dropna(inplace=True)
    
    @staticmethod
    def estimate_best_hyperparameters(X, y):
        """
        Estimate the best hyperparameters for the XGBoost model.
        """
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.05]
        }
        tscv = TimeSeriesSplit(n_splits=5)
        grid = GridSearchCV(
            XGBRegressor(random_state=42),
            param_grid, 
            cv=tscv,
            scoring='neg_root_mean_squared_error', 
            verbose=0, 
            n_jobs=-1
        )
        grid.fit(X, y)
        return grid.best_params_

    @staticmethod
    def evaluate_with_timeseries_cv(X, y_high, y_low):
        tscv = TimeSeriesSplit(n_splits=5)
        high_rmse_scores = []
        low_rmse_scores = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_high_train, y_high_test = y_high.iloc[train_idx], y_high.iloc[test_idx]
            y_low_train, y_low_test = y_low.iloc[train_idx], y_low.iloc[test_idx]

            xgb_high = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, random_state=42)
            xgb_low = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, random_state=42)

            xgb_high.fit(X_train, y_high_train, eval_set=[(X_test, y_high_test)], verbose=False)
            xgb_low.fit(X_train, y_low_train, eval_set=[(X_test, y_low_test)], verbose=False)

            y_high_pred = xgb_high.predict(X_test)
            y_low_pred = xgb_low.predict(X_test)

            rmse_high = np.sqrt(mean_squared_error(y_high_test, y_high_pred))
            rmse_low = np.sqrt(mean_squared_error(y_low_test, y_low_pred))

            high_rmse_scores.append(rmse_high)
            low_rmse_scores.append(rmse_low)

            logger.debug(f"\n--- Fold {fold + 1} ---")
            logger.debug(f"Resistance RMSE: {rmse_high:.4f} | Support RMSE: {rmse_low:.4f}")

        logger.debug("\n--- Average Results ---")
        logger.debug(f"Average Resistance RMSE: {np.mean(high_rmse_scores):.4f}")
        logger.debug(f"Average Support RMSE: {np.mean(low_rmse_scores):.4f}")

    def train_xgb_models(self):
        features = ["close", "rsi", "atr", "ema20", "ema50", "bb_high", "bb_low"]
        X = self.df_final[features]
        # Targets in log-space
        y_high_log = self.df_final["future_max_log"]
        y_low_log = self.df_final["future_min_log"]

        # self.evaluate_with_timeseries_cv(X, y_high, y_low)

        logger.debug("\n--- Estimating Best Hyperparameters ---")
        best_params_high = self.estimate_best_hyperparameters(X, y_high_log)
        best_params_low = self.estimate_best_hyperparameters(X, y_low_log)
        logger.debug(f"Best Params (Resistance): {best_params_high}")
        logger.debug(f"Best Params (Support): {best_params_low}")

        xgb_high_final = XGBRegressor(**best_params_high, random_state=42)
        xgb_low_final = XGBRegressor(**best_params_low, random_state=42)

        xgb_high_final.fit(X, y_high_log)
        xgb_low_final.fit(X, y_low_log)

        latest_X = X.iloc[[-1]]
        latest_resistance_log = xgb_high_final.predict(latest_X)[0]
        latest_support_log = xgb_low_final.predict(latest_X)[0]

        # Convert back to original price scale
        latest_resistance = np.exp(latest_resistance_log)
        latest_support = np.exp(latest_support_log)

        # Calculate prediction confidence based on model uncertainty
        # Notice that it is assumed that when atr == prediction_range
        # the confidence is 80%, then the scaling factor would be 20.
        atr_value = latest_X["atr"].iloc[-1]
        prediction_range = latest_resistance - latest_support
        confidence = min(100, max(0, 100 - (prediction_range / atr_value) * 20))

        smart_round = lambda x: float("{:.4g}".format(x) if abs(x) < 1 else "{:.4f}".format(x))

        return {
            "message": "Final Model Live Prediction",
            "time_frame": self.timeframe,
            "prediction_confidence": round(confidence, 1),
            "latest_predicted_resistance": smart_round(latest_resistance),
            "latest_predicted_support": smart_round(latest_support)
        }


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
            logger.error(f"[ERROR] Unable to fetch market data: {e}")
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

###############################################################################
# Utility functions for miscellaneous or supporting tasks unrelated to the core
# business logic. These may include helpers for formatting, validation, or
# other general-purpose operations used throughout the codebase.
###############################################################################

def format_message_spikes(*args: Dict[str, Any]) -> str:
    """
    Formats message data from multiple dictionaries, filtering out messages
    where both price changes are less than MIN_PRICE_CHANGE and volume change 
    is less than MIN_VOLUME_CHANGE.

    Args:
        *args: Variable number of dictionaries containing message data.
               Each dictionary should have the keys:
               - 'symbol', 'price_rate', 'volume_rate', 'atr_pct', 'close'

    Returns:
        str: A formatted string containing all messages that meet the
             filtering criteria.
    """ 
    messages = []

    def safe_float(raw):
        try:
            return float(raw.get("price_rate", 0))
        except (TypeError, ValueError):
            return 0.0

    for raw in sorted(args, key=safe_float, reverse=True):
        try:   
            price_rate = float(raw.get('price_rate', 0))
            volume_rate = float(raw.get('volume_rate', 0))
            atr_pct = float(raw.get('atr_pct', 0))
            close = float(raw.get('close', 0))
        except ValueError:
            continue

        if abs(price_rate) < MIN_PRICE_CHANGE and volume_rate < MIN_VOLUME_CHANGE:
            continue

        # Build message using f-string
        messages.append(
            (
                f"\nSymbol: {raw.get('symbol', 'N/A')}\n"
                f"Price Change: {price_rate:.2f}%\n"
                f"Volume Change: {volume_rate:.2f}%\n"
                f"ATR Percentage: {atr_pct:.2f}%\n"
                f"Close Price: {close}\n"
                f"──────────────"
            )
        )

    return "\n".join(messages)


def format_symbol_name(symbol: str) -> str:
    """Format symbol name for trading."""
    if re.match(r"^[^/\s\d]*", symbol, re.IGNORECASE):
        return f'{symbol.upper()}/USDT:USDT'
    return ''


def calculate_correlation(
    df: pd.DataFrame, 
    measure_column: str = "atr", 
    nlags: int = 200, 
    threshold: float = 0.1,
    min_window: int = 5,
    default_window: int = 50
) -> int:
    """
    Calculate the optimal window size for ATR averaging based on autocorrelation.

    Args:
        df: DataFrame containing the measure to calculate the correlation.
        measure_column: The column name to analyze.
        nlags: Number of lags to compute in autocorrelation.
        threshold: The absolute autocorrelation value below which to consider the lag uncorrelated.
        min_window: Minimum window size to return.
        default_window: Default window size if no lag meets the threshold.

    Returns:
        int: Suggested window size based on autocorrelation analysis.
    """
    if measure_column not in df.columns:
        msg = (
            f"Column '{measure_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
        logger.error(msg)
        raise ValueError(msg)

    atr_series = df[measure_column].dropna()
    if len(atr_series) < min_window:
        logger.info(
            "Not enough data to compute autocorrelation "
            f"(need at least {min_window} values). Returning min_window={min_window}."
        )
        return min_window

    acf_vals = acf(atr_series, nlags=nlags, fft=True)

    # Skip lag 0 (always 1.0), start from lag 1
    for lag, val in enumerate(acf_vals[1:], start=1):
        if abs(val) < threshold:
            logger.info(
                f"Suggested window: {lag} (autocorrelation dropped below {threshold})"
            )
            return max(lag, min_window)

    logger.info(
        f"No lag found with autocorrelation below {threshold}. "
        f"Returning default_window={default_window}."
    )
    return default_window
