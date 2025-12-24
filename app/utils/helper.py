import asyncio
import re
import time
import traceback
from concurrent.futures.thread import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Self, Tuple, TypeVar

import numpy as np
import pandas as pd
import requests
import talib as ta
from fastapi import Query
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from statsmodels.tsa.stattools import acf
from xgboost import XGBRegressor

import ccxt.async_support as ccxt_async

from app.core.logging import AppLogger

# Initialize logging
logger = AppLogger.get_logger()

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed. Using GridSearchCV. Install with: pip install optuna")

MIN_PRICE_CHANGE = 3
MIN_VOLUME_CHANGE = 5000 # Notice that the volumen movements are above 100 then 5000 is a good threshold.
## The time limit should be calcuated dimanically based on the timeframe.
## For example, if the timeframe is 15m, the limit should be 96.
## If the timeframe is 1h, the limit should be 24.
## If the timeframe is 4h, the limit should be 6.
## If the timeframe is 1d, the limit should be 1.
SINCE_24H_AGO_LIMIT = 96
_INDICATOR_EXECUTOR = ThreadPoolExecutor(max_workers=4)
T = TypeVar("T")


class IndicatorComputer:
    """Computes technical indicators on linear or log-transformed prices."""

    def __init__(self, df: pd.DataFrame):
        self._df_transformed = df.copy()
        self.close = self._df_transformed["close"].values
        self.high = self._df_transformed["high"].values
        self.low = self._df_transformed["low"].values
        self.volume = self._df_transformed["volume"].values

    def get_df_transformed(self) -> pd.DataFrame:
        return self._df_transformed

    def compute_atr(self, window: int = 14) -> Self:
        self._df_transformed["atr"] = ta.ATR(self.high, self.low, self.close, timeperiod=window)
        self._df_transformed["atr_pct"] = (self._df_transformed["atr"] / np.abs(self.close)) * 100
        return self

    def compute_atr_above_mean(self) -> Self:
        if "atr" not in self._df_transformed:
            msg = "ATR not found in DataFrame. Please call compute_atr() first."
            logger.error(msg)
            raise ValueError(msg)

        median_window = calculate_correlation(self._df_transformed)

        if len(self._df_transformed) <= median_window:
            self._df_transformed['atr_mean'] = 0.0
            self._df_transformed['atr_above_mean'] = False
            logger.info("Not enough data to compute ATR median.")
        else:
            self._df_transformed['atr_mean'] = self._df_transformed['atr'].rolling(window=median_window).mean()
            self._df_transformed['atr_above_mean'] = self._df_transformed['atr'] > self._df_transformed['atr_mean']
        return self

    def compute_bbands(self, window: int = 20) -> Self:
        bb_upper, bb_middle, bb_lower = ta.BBANDS(self.close, timeperiod=window, nbdevup=2, nbdevdn=2)
        self._df_transformed["bb_position"] = (self.close - bb_lower) / (bb_upper - bb_lower + 1e-10)
        self._df_transformed["bb_width_pct"] = ((bb_upper - bb_lower) / bb_middle) * 100
        self._df_transformed["bb_upper_dist"] = ((bb_upper - self.close) / self.close) * 100
        self._df_transformed["bb_lower_dist"] = ((bb_lower - self.close) / self.close) * 100
        return self

    def compute_rsi(self, window: int = 14) -> Self:
        self._df_transformed["rsi"] = ta.RSI(self.close, timeperiod=window)
        return self

    def compute_roc(self, metric: str = "price", window: int = 10) -> Self:
        if metric == "price":
            metric_values = self.close
        elif metric == "volume":
            metric_values = self.volume
        else:
            msg = f"Unsupported metric '{metric}'. Expected 'price' or 'volume'."
            logger.error(msg)
            raise ValueError(msg)
        
        if window == 1: # The rate is the same as the ROC for window 1
            self._df_transformed[f"{metric}_rate"] = ta.ROC(metric_values, timeperiod=window)
        else:
            self._df_transformed[f"{metric}_rate{window}"] = ta.ROC(metric_values, timeperiod=window)
        return self

    def compute_macd(self, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> Self:
        macd, macd_signal, macd_hist = ta.MACD(self.close, fastperiod, slowperiod, signalperiod)
        self._df_transformed["macd"], self._df_transformed["macd_signal"], self._df_transformed["macd_hist"] = macd, macd_signal, macd_hist
        return self

    def compute_stochastic(self, window: int = 14) -> Self:
        self._df_transformed["stochastic"] = ta.STOCH(self.high, self.low, self.close, fastk_period=window, slowk_period=window, slowd_period=window)
        return self

    def compute_ema(self, window: int = 20) -> Self:
        self._df_transformed[f"ema{window}"] = ta.EMA(self.close, timeperiod=window)
        return self

    def compute_volume_indicators(self) -> Self:
        self._df_transformed["volume_sma20"] = self._df_transformed["volume"].rolling(20).mean()
        self._df_transformed["volume_ratio"] = self._df_transformed["volume"] / (self._df_transformed["volume_sma20"] + 1e-10)
        self._df_transformed["volume_zscore"] = (self._df_transformed["volume"] - self._df_transformed["volume"].rolling(50).mean()) / (self._df_transformed["volume"].rolling(50).std() + 1e-10)
        
        return self

    def compute_obv(self) -> Self:
        self._df_transformed["obv"] = ta.OBV(self.close, self.volume)
        self._df_transformed["obv_ema"] = self._df_transformed["obv"].ewm(span=20).mean()
        self._df_transformed["obv_slope"] = self._df_transformed["obv"].pct_change(5)
        return self

    async def run_in_thread(self, func: Callable[[], T]) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_INDICATOR_EXECUTOR, func)

    async def compute_async(self) -> Self:
        def compute_all() -> Self:
            return (
                self.compute_atr()
                .compute_bbands()
                .compute_rsi()
                .compute_roc(metric="price")
                .compute_roc(metric="volume")
                .compute_macd()
                .compute_ema()
                .compute_volume_indicators()
            )

        return await self.run_in_thread(compute_all)

class BaseAnalyzer:
    """Base class for volume analysis with technical indicators."""
    
    def __init__(
            self, 
            exchange_id: str = "binance", 
            ) -> None:
        self.exchange = None
        self.exchange_id = exchange_id
        self.indicator_computer = IndicatorComputer


    async def initialize(self) -> None:
        """
        Initialize the Binance futures exchange.
        """

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
        """
        Properly close the exchange connection and underlying aiohttp session.
        """
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

    async def get_futures_pairs(self, pair:str = 'USDT') -> Dict[str, Any]:
        """
        Fetch all future tickets pairs.
        
        Raises:
            RuntimeError: If exchange is not initialized.
        """
        if not self.exchange:
            raise RuntimeError("Exchange not initialized. Call initialize() first.")

        data = await self.exchange.fetch_tickers()
        
        pairs = [s for s in data.keys() if s.endswith(pair)]
        
        if not pairs:
            msg = f"No {pair} pairs found."
            logger.warning(msg)
            raise RuntimeWarning(msg)
            
        return pairs

    async def get_historical_data(self, symbol: str, timeframe: str = '15m', limit: int = SINCE_24H_AGO_LIMIT) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single timeframe — stateless.
        
        Raises:
            RuntimeError: If exchange is not initialized.
        """

        ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['event_timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df


class BinanceVolumeAnalyzer(BaseAnalyzer):
    """Extended volume analyzer for Binance with market spike detection."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._df_final_values = pd.DataFrame()

    async def process_symbol(
        self,
        symbol: str,
        timeframe: str = '15m',
        limit: int = SINCE_24H_AGO_LIMIT,
        window: int = 1
    ) -> pd.DataFrame:
        """
        Process individual symbol data.
        """
        df = await self.get_historical_data(symbol, timeframe, limit)

        if df is not None and len(df) > 1:
            indicator_computer = self.indicator_computer(df)
            await indicator_computer.run_in_thread(
                lambda: indicator_computer.compute_atr().compute_roc(metric="price", window=window)
            )
            df_transformed = indicator_computer.get_df_transformed()
            match = re.match(r"^[^/\s]*", symbol)
            df_transformed["symbol"] = match.group(0) if match else symbol
            return df_transformed.iloc[[-1]]


        return pd.DataFrame()

    async def calculate_market_spikes(
        self,
        timeframe: str = '15m',
        limit: int = SINCE_24H_AGO_LIMIT,
        max_concurrency: int = 100
        ) -> None:
        """
        Process all USDT futures pairs, calculate price and volume changes, 
        and aggregate the latest results. Raises if no significant changes found.
        """

        # --- 1. Validation & Setup ---
        if not self.exchange:
            raise RuntimeError("Exchange not initialized.")
            
        usdt_pairs = await self.get_futures_pairs()

        # --- 2. The Clean Worker ---
        # We define the worker here to capture 'semaphore', 'timeframe', etc.
        semaphore = asyncio.Semaphore(max_concurrency)

        async def fetch_pair(symbol: str) -> None:
            async with semaphore:
                try:
                    # Try to get the dataframe
                    df = await self.process_symbol(symbol, timeframe, limit)
                    # Return it only if it has data
                    return df if not df.empty else None
                except Exception as e:
                    # Log immediately while we still have the 'symbol' context
                    logger.warning(f"[{symbol}] Task failed: {e}")
                    logger.debug(traceback.format_exc())
                    return None

        # --- 3. Execution ---
        # We run all tasks. If they fail, they return None (safely).
        tasks = [fetch_pair(symbol) for symbol in usdt_pairs]
        results = await asyncio.gather(*tasks)

        # --- 4. Final Aggregation ---
        # Filter out the Nones in one clean line
        valid_dfs = [res for res in results if res is not None]

        if not valid_dfs:
            logger.info("No significant price rate at the moment")
            raise RuntimeWarning("No significant price rate")

        self._df_final_values = pd.concat(valid_dfs, ignore_index=True)


    def get_top_symbols(
        self, 
        metric: str = "price_rate", 
        ascending: bool = False,
        n_values: int = 3,
        threshold: int = 1
    ) -> pd.DataFrame:
        """
        Return a DataFrame with the top symbols ranked by the specified metric.

        - Handles missing or invalid metric names gracefully by raising ValueError.
        - Does not mutate the structure of the resulting DataFrame in a way that would break a list-of-dict contract.
        - Selects only the columns ['symbol', 'event_timestamp', 'price_rate', 'atr_pct', 'close'] for output.
        - Results are sorted by the metric, in ascending or descending order, and limited to the top n_values.
        """
        if self._df_final_values.empty:
            logger.warning("DataFrame is empty. Please call calculate_market_spikes first.")
            raise ValueError("DataFrame is empty. Please call calculate_market_spikes first.")

        if metric not in ["price_rate", "volume_rate"]:
            msg = f"Unsupported metric '{metric}'. Expected 'price_rate'."
            logger.error(msg)
            raise ValueError(msg)

        df_sorted = (
            self._df_final_values[['symbol', 'event_timestamp', 'price_rate', 'atr_pct', 'close']].copy()
            .loc[lambda x: abs(x['price_rate']) > threshold]
            .sort_values(by=metric, ascending=ascending)
            .head(n_values)
            .reset_index(drop=True)
        )

        return df_sorted


class XGBoostSupportResistancePredictor(BaseAnalyzer):
    """XGBoostSupportResistancePredictor uses XGBoost machine learning to predict 
    cryptocurrency support and resistance levels using technical indicators."""

    def __init__(self, window: int = 10, n_splits: int = 5, tune_hyperparams: bool = True, use_optuna: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.df_final = pd.DataFrame()
        self.window = window
        self.n_splits = n_splits
        self.tune_hyperparams = tune_hyperparams
        self.use_optuna = use_optuna and OPTUNA_AVAILABLE  # Fall back to GridSearch if Optuna unavailable
        self.model_high = None
        self.model_low = None
        self.feature_cols = None
        self.variance_selector = None
        self.corr_features_to_drop = []
        self.best_params_high = None
        self.best_params_low = None

    async def get_historical_data(self, symbol: str, timeframe: str = '15m', limit: int = SINCE_24H_AGO_LIMIT) -> pd.DataFrame:
        df = await super().get_historical_data(symbol, timeframe, limit)
        if not df.empty:
            df.set_index("timestamp", inplace=True)
            return df
        return pd.DataFrame()
 
    async def add_features(self, df: pd.DataFrame, fast_mode: bool = True) -> Self:
        """
        Add technical indicators to the dataframe.
        
        """
        indicator_object = self.indicator_computer(df)
        await indicator_object.compute_async()
        self.df_final = indicator_object.get_df_transformed()

        # # === MOMENTUM FEATURES ===
        self.df_final["rsi_ma"] = self.df_final["rsi"].rolling(5).mean()
        self.df_final["rsi_slope"] = self.df_final["rsi"].diff(3)
        self.df_final["roc"] = ta.ROC(self.df_final["close"], timeperiod=10)
        self.df_final["roc_5"] = ta.ROC(self.df_final["close"], timeperiod=5)
        self.df_final["roc_20"] = ta.ROC(self.df_final["close"], timeperiod=20)

        # MACD (normalized by price)
        self.df_final["macd_pct"] = (self.df_final["macd"] / self.df_final["close"]) * 100
        self.df_final["macd_signal_pct"] = (self.df_final["macd_signal"] / self.df_final["close"]) * 100
        self.df_final["macd_hist_pct"] = (self.df_final["macd_hist"] / self.df_final["close"]) * 100

        # Stochastic
        slowk, slowd = ta.STOCH(
            self.df_final["high"], 
            self.df_final["low"], 
            self.df_final["close"], 
            fastk_period=14, 
            slowk_period=3, 
            slowd_period=3
            )
        self.df_final["stoch_k"] = slowk
        self.df_final["stoch_d"] = slowd

        # === TREND FEATURES ===
        self.df_final["ema5"] = ta.EMA(self.df_final["close"], timeperiod=5)
        self.df_final["ema10"] = ta.EMA(self.df_final["close"], timeperiod=10)
        self.df_final["ema50"] = ta.EMA(self.df_final["close"], timeperiod=50)
        
        self.df_final["ema5_dist"] = ((self.df_final["close"] - self.df_final["ema5"]) / self.df_final["close"]) * 100
        self.df_final["ema10_dist"] = ((self.df_final["close"] - self.df_final["ema10"]) / self.df_final["close"]) * 100
        self.df_final["ema20_dist"] = ((self.df_final["close"] - self.df_final["ema20"]) / self.df_final["close"]) * 100
        self.df_final["ema50_dist"] = ((self.df_final["close"] - self.df_final["ema50"]) / self.df_final["close"]) * 100
        
        # EMA slopes
        self.df_final["ema10_slope"] = self.df_final["ema10"].pct_change(5) * 100
        self.df_final["ema20_slope"] = self.df_final["ema20"].pct_change(5) * 100
        
        # EMA crossovers
        self.df_final["ema5_10_cross"] = ((self.df_final["ema5"] - self.df_final["ema10"]) / self.df_final["close"]) * 100
        self.df_final["ema10_20_cross"] = ((self.df_final["ema10"] - self.df_final["ema20"]) / self.df_final["close"]) * 100
        
        # Trend strength
        self.df_final["adx"] = ta.ADX(self.df_final["high"], self.df_final["low"], self.df_final["close"], timeperiod=14)
        self.df_final["plus_di"] = ta.PLUS_DI(self.df_final["high"], self.df_final["low"], self.df_final["close"], timeperiod=14)
        self.df_final["minus_di"] = ta.MINUS_DI(self.df_final["high"], self.df_final["low"], self.df_final["close"], timeperiod=14)

        # === REGIME FEATURES ===
        if not fast_mode:
            # SLOW version: More accurate but 10x slower
            self.df_final["volatility_regime"] = self.df_final["atr_pct"].rolling(100).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
            )
            
            self.df_final["volume_regime"] = self.df_final["volume_ratio"].rolling(50).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
            )
        else:
            # FAST version: Approximate with percentile (100x faster)
            # This gives similar information without expensive .apply()
            self.df_final["volatility_regime"] = self.df_final["atr_pct"].rolling(100).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) if len(x) > 0 else 0.5,
                raw=False
            )
            
            self.df_final["volume_regime"] = self.df_final["volume_ratio"].rolling(50).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) if len(x) > 0 else 0.5,
                raw=False
            )
        
        # Price position in recent range (vectorized - fast)
        self.df_final["price_position_50"] = (
            self.df_final["close"] - self.df_final["low"].rolling(50).min()
        ) / (
            self.df_final["high"].rolling(50).max()
            - self.df_final["low"].rolling(50).min()
            + 1e-10
        )
        self.df_final.dropna(inplace=True)
        
        return self
    
    def generate_targets(self) -> Self:
        """
        Generate forward-looking S/R targets WITHOUT lookahead bias.
        Target: % distance from current close to future max/min.
        """

        df = self.df_final.copy()
        # CRITICAL: Shift AFTER computing rolling aggregates
        df["future_high"] = df["high"].rolling(self.window, min_periods=1).max().shift(-self.window)
        df["future_low"] = df["low"].rolling(self.window, min_periods=1).min().shift(-self.window)
        
        # Convert to % distance from current price (scale-invariant)
        df["target_resistance_pct"] = ((df["future_high"] - df["close"]) / df["close"]) * 100
        df["target_support_pct"] = ((df["future_low"] - df["close"]) / df["close"]) * 100
        
        # Remove last N rows (no targets available)
        df = df.iloc[:-self.window]
        
        return df.dropna()

    def select_features(self, X_train, X_test, variance_threshold=0.01, correlation_threshold=0.85):
        """
        Feature selection using variance threshold and correlation filtering.
        """
        
        # We drop ROWS (axis=0) with NaNs to remove the "warm-up" period.
        # This allows us to calculate the true variance of the indicators.
        X_train_clean = X_train.dropna(axis=0)
        
        # Safety check: If dropping rows removes everything (e.g. not enough history),
        # fallback to fillna(0) so the code doesn't crash.
        if len(X_train_clean) < 10:
            logger.warning("Too many NaNs, falling back to fillna(0) for selection.")
            X_train_clean = X_train.fillna(0)

        # 1. VARIANCE THRESHOLD
        if self.variance_selector is None:
            self.variance_selector = VarianceThreshold(threshold=variance_threshold)
            self.variance_selector.fit(X_train_clean) # Fit on clean data
        
        # Get the list of columns that survived the threshold
        variance_mask = self.variance_selector.get_support()
        selected_after_variance = X_train.columns[variance_mask].tolist()
        
        logger.info("After variance filter: %s features", len(selected_after_variance))
        
        # We manually select columns instead of using .transform()
        # This avoids dimension mismatch errors and preserves the original dataframe structure.
        X_train_var = X_train[selected_after_variance]
        X_test_var = X_test[selected_after_variance]

        # 2. CORRELATION FILTERING
        if not self.corr_features_to_drop:
            # Calculate correlation only on clean data to avoid noise
            # (Again, we drop warm-up rows just for this calculation)
            corr_matrix = X_train_var.dropna(axis=0).corr().abs()
            
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            self.corr_features_to_drop = [
                column for column in upper.columns 
                if any(upper[column] > correlation_threshold)
            ]
            
            if self.corr_features_to_drop:
                logger.info("Removing %s highly correlated features", len(self.corr_features_to_drop))
        
        # Drop the correlated features
        X_train_final = X_train_var.drop(columns=self.corr_features_to_drop, errors='ignore')
        X_test_final = X_test_var.drop(columns=self.corr_features_to_drop, errors='ignore')
        
        final_features = X_train_final.columns.tolist()
        logger.info("Final feature count: %s", len(final_features))
        
        return X_train_final, X_test_final, final_features
    
    def tune_hyperparameters_optuna(self, X_train, y_train, model_name="Model", n_trials=50):
        """
        Use Optuna to find best hyperparameters (FASTER and BETTER than GridSearchCV).
        Uses Bayesian optimization with early stopping.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_name: Name for logging
            n_trials: Number of optimization trials (default 50, vs 324 for GridSearch)
        
        Returns:
            dict with best parameters
        """
        # Set Optuna to be quiet
        if OPTUNA_AVAILABLE:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        logger.info("Tuning hyperparameters for %s using Optuna...", model_name)
        logger.info("Running %s trials (GridSearch would run 324)...", n_trials)

        # Pre-define split indices to save time inside the loop
        tscv = TimeSeriesSplit(n_splits=3)
        # Generate indices once
        cv_indices = list(tscv.split(X_train))
        
        # Define objective function
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            }
            
            scores = []
            
            for train_idx, val_idx in cv_indices:
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model = XGBRegressor(**params, random_state=42, n_jobs=1)
                model.fit(X_tr, y_tr, verbose=False)
                
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                scores.append(rmse)
                
                # Report intermediate value for pruning
                trial.report(rmse, len(scores))
                
                # Prune unpromising trials
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return np.mean(scores)
        
        # Create study with pruning
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=1)
        
        logger.info("Best RMSE: %.4f", study.best_value)
        logger.info("Best params: %s", study.best_params)
        logger.info(
            "Completed %s trials (%s pruned)",
            len(study.trials),
            len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        )
        
        return study.best_params

    def tune_hyperparameters(self, X_train, y_train, model_name="Model"):
        """
        Use GridSearchCV to find best hyperparameters (fallback if Optuna not available).
        Uses TimeSeriesSplit to respect temporal order.
        """
        logger.info("Tuning hyperparameters for %s using GridSearchCV...", model_name)
        
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.05, 0.1],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        # Use time-series CV for hyperparameter tuning
        tscv = TimeSeriesSplit(n_splits=3)
        
        base_model = XGBRegressor(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info("Best params: %s", grid_search.best_params_)
        logger.info("Best CV RMSE: %.4f", -grid_search.best_score_)
        
        return grid_search.best_params_

    async def train(self, df, refit_features_each_fold=False, xgb_callbacks=None):
        """
        Train models with proper time-series cross-validation.
        Includes feature selection and hyperparameter tuning.
        
        Args:
            df: DataFrame with OHLCV data
            refit_features_each_fold: If True, recalculate feature selection per fold.
            xgb_callbacks: Optional list of XGBoost callbacks.
        """
        # Add features and targets
        await self.add_features(df, fast_mode=True)  # Use accurate mode for training
        df = self.generate_targets()
   
        # Get all possible features
        all_feature_cols = [col for col in df.columns if col not in [
            'event_timestamp', 'open', 'high', 'low', 'close', 'volume',
            'future_high', 'future_low', 'target_resistance_pct', 'target_support_pct',
            'ema5', 'ema10', 'ema20', 'ema50'  # Exclude intermediate EMAs, keep only _dist versions
        ]]
        
        X = df[all_feature_cols].copy()
        y_high = df["target_resistance_pct"]
        y_low = df["target_support_pct"]
        
        logger.info("Training on %s samples", len(df))
        logger.info("Initial feature count: %s", len(all_feature_cols))
        logger.info("Predicting %s-candle forward S/R levels", self.window)

        def _train_sync():
            # Time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            
            high_scores = []
            low_scores = []
            
            # Reset feature selection state
            self.variance_selector = None
            self.corr_features_to_drop = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                logger.info("%s", "=" * 60)
                logger.info("Fold %s/%s", fold + 1, self.n_splits)
                logger.info("%s", "=" * 60)
                
                X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
                y_high_train, y_high_test = y_high.iloc[train_idx], y_high.iloc[test_idx]
                y_low_train, y_low_test = y_low.iloc[train_idx], y_low.iloc[test_idx]
                
                # Feature selection logic:
                if refit_features_each_fold:
                    if fold > 0:
                        self.variance_selector = None
                        self.corr_features_to_drop = []
                    
                    X_train, X_test, selected_features = self.select_features(
                        X_train_raw, X_test_raw,
                        variance_threshold=0.01,
                        correlation_threshold=0.85
                    )
                    self.feature_cols = selected_features
                
                elif fold == 0:
                    X_train, X_test, selected_features = self.select_features(
                        X_train_raw, X_test_raw,
                        variance_threshold=0.01,
                        correlation_threshold=0.85
                    )
                    self.feature_cols = selected_features
                
                else:
                    X_train = X_train_raw[self.feature_cols].fillna(0)
                    X_test = X_test_raw[self.feature_cols].fillna(0)
                
                # Hyperparameter tuning (only on first fold to save time)
                if fold == 0 and self.tune_hyperparams:               
                    if self.use_optuna:
                        logger.info("HYPERPARAMETER TUNING with OPTUNA (using first fold)")
                        logger.info("%s", "=" * 60)
                        self.best_params_high = self.tune_hyperparameters_optuna(
                            X_train, y_high_train, "Resistance Model", n_trials=50
                        )
                        self.best_params_low = self.tune_hyperparameters_optuna(
                            X_train, y_low_train, "Support Model", n_trials=50
                        )
                    else:
                        logger.info("HYPERPARAMETER TUNING with GRIDSEARCH (using first fold)")
                        logger.info("%s", "=" * 60)
                        self.best_params_high = self.tune_hyperparameters(
                            X_train, y_high_train, "Resistance Model"
                        )
                        self.best_params_low = self.tune_hyperparameters(
                            X_train, y_low_train, "Support Model"
                        )
                elif fold == 0 and not self.tune_hyperparams:
                    # Use default good parameters
                    self.best_params_high = {
                        'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05,
                        'min_child_weight': 3, 'subsample': 0.8, 'colsample_bytree': 0.8
                    }
                    self.best_params_low = self.best_params_high.copy()
                
                # Train models with best parameters
                if xgb_callbacks:
                    self.best_params_high['callbacks'] = xgb_callbacks
                    self.best_params_low['callbacks'] = xgb_callbacks

                xgb_high = XGBRegressor(**self.best_params_high, random_state=42)
                xgb_low = XGBRegressor(**self.best_params_low, random_state=42)
                
                xgb_high.fit(X_train, y_high_train)
                xgb_low.fit(X_train, y_low_train)
                
                # Evaluate
                y_high_pred = xgb_high.predict(X_test)
                y_low_pred = xgb_low.predict(X_test)
                
                # Baseline (predict mean)
                baseline_high_pred = np.full_like(y_high_test, y_high_train.mean())
                baseline_low_pred = np.full_like(y_low_test, y_low_train.mean())
                
                # Metrics
                rmse_high = np.sqrt(mean_squared_error(y_high_test, y_high_pred))
                rmse_low = np.sqrt(mean_squared_error(y_low_test, y_low_pred))
                
                r2_high = r2_score(y_high_test, y_high_pred)
                r2_low = r2_score(y_low_test, y_low_pred)
                
                baseline_r2_high = r2_score(y_high_test, baseline_high_pred)
                baseline_r2_low = r2_score(y_low_test, baseline_low_pred)
                
                high_scores.append((rmse_high, r2_high))
                low_scores.append((rmse_low, r2_low))
                
                logger.info(
                    "Fold %s Results: Resistance RMSE=%.4f, R²=%.4f (baseline R²=%.4f)",
                    fold + 1,
                    rmse_high,
                    r2_high,
                    baseline_r2_high,
                )
                logger.info(
                    "Fold %s Results: Support RMSE=%.4f, R²=%.4f (baseline R²=%.4f)",
                    fold + 1,
                    rmse_low,
                    r2_low,
                    baseline_r2_low,
                )
                
                # Feature importance (only for first fold)
                if fold == 0:
                    importance_high = pd.DataFrame({
                        'feature': selected_features,
                        'importance': xgb_high.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    logger.info("Top 10 Features for Resistance:\n%s", importance_high.head(10).to_string(index=False))
            
            # Print average scores
            logger.info("%s", "=" * 60)
            logger.info("AVERAGE CROSS-VALIDATION RESULTS")
            logger.info("%s", "=" * 60)
            
            avg_rmse_high = np.mean([s[0] for s in high_scores])
            avg_r2_high = np.mean([s[1] for s in high_scores])
            avg_rmse_low = np.mean([s[0] for s in low_scores])
            avg_r2_low = np.mean([s[1] for s in low_scores])
            
            logger.info("Resistance: RMSE=%.4f, R²=%.4f", avg_rmse_high, avg_r2_high)
            logger.info("Support: RMSE=%.4f, R²=%.4f", avg_rmse_low, avg_r2_low)
            
            # Train final models on ALL data with selected features
            logger.info("%s", "=" * 60)
            logger.info("TRAINING FINAL MODELS ON FULL DATASET")
            logger.info("%s", "=" * 60)
            
            X_full = X[self.feature_cols].copy()
            
            self.model_high = XGBRegressor(**self.best_params_high, random_state=42)
            self.model_low = XGBRegressor(**self.best_params_low, random_state=42)
            
            self.model_high.fit(X_full, y_high)
            self.model_low.fit(X_full, y_low)
            
            logger.info("Final models trained with %s features", len(self.feature_cols))

        # Run training in thread to avoid blocking asyncio loop
        await asyncio.to_thread(_train_sync)
        
        return self
    
    async def predict_levels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        Returns absolute price levels.
        
        Args:
            df: DataFrame with OHLCV data
        """
        # Add features (this may drop some rows due to NaN from indicators)
        df_feat = df.copy()
        # Use only the selected features
        X = df_feat[self.feature_cols].copy()
        
        # Predict % distances (XGBoost prediction is fast)
        resistance_pct = self.model_high.predict(X.fillna(0))
        support_pct = self.model_low.predict(X.fillna(0))
        
        # Get corresponding close prices (aligned with X after dropna)
        current_prices = df_feat["close"].values
        
        # Convert back to absolute prices
        resistance_levels = current_prices * (1 + resistance_pct / 100)
        support_levels = current_prices * (1 + support_pct / 100)
        
        return resistance_levels, resistance_pct, support_levels, support_pct

    async def predict_latest(self, lookback=200):
        """
        Predict S/R levels for the most recent candle.
        OPTIMIZED for speed in live trading.
        
        NOTE: This method uses `self.df_final` so you must ensure `add_features` 
        was called recently with up-to-date data.
        
        Args:
            lookback: Number of historical candles to use (need enough for indicators)
  
        Returns:
            dict with current price, resistance, support, and risk/reward ratio
        """
        # Keep strictly if needed for performance metrics, otherwise move to top
        start_time = time.time()
        
        if self.df_final.empty:
            logger.warning("df_final is empty. Cannot predict.")
            return None

        # Take recent data for indicator calculation
        # This fetches the last `lookback` rows in an efficient, vectorized manner:
        recent_df = self.df_final.tail(lookback)
        
        # Get predictions
        resistance, resistance_pct, support, support_pct = await self.predict_levels(recent_df)
        
        if len(resistance) == 0:
            return None
        
        # Latest values
        latest_close = recent_df["close"].iloc[-1]
        latest_resistance = resistance[-1]
        latest_support = support[-1]
        latest_resistance_pct = resistance_pct[-1]
        latest_support_pct = support_pct[-1]
        
        # Calculate metrics
        risk_reward = abs(latest_resistance_pct / latest_support_pct) if latest_support_pct != 0 else 0
        
        elapsed_time = time.time() - start_time

        smart_round = lambda x: float("{:.4g}".format(x) if abs(x) < 1 else "{:.4f}".format(x))
        
        return {
            "current_price": latest_close,
            "resistance": smart_round(latest_resistance),
            "support":smart_round(latest_support),
            "upside_pct": round(float(latest_resistance_pct), 2),
            "downside_pct": round(float(latest_support_pct), 2),
            "risk_reward_ratio": risk_reward,
            "timestamp": recent_df.index[-1],
            "prediction_time_ms": elapsed_time * 1000  # Convert to milliseconds
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

def format_message_events(*args: Dict[str, Any]) -> str:
    """
    Formats message data from multiple dictionaries, filtering out messages
    where the price changes are less than MIN_PRICE_CHANGE.

    Args:
        *args: Variable number of dictionaries containing message data.
               Each dictionary should have the keys:
               - 'symbol', 'price_rate', 'atr_pct', 'close'

    Returns:
        str: A formatted string containing all messages that meet the
             filtering criteria.
    """ 
    messages = []

    for raw in args:
        # Build message using f-string
        try:
            messages.append(
                (
                    f"\nSymbol: {raw.get('symbol', 'N/A')}\n"
                    f"Price Change: {raw.get('price_rate', 0):.2f}%\n"
                    f"ATR Percentage: {raw.get('atr_pct', 0):.2f}%\n"
                    f"Close Price: {raw.get('close', 0)}\n"
                    f"──────────────"
                )
            )
        except ValueError as e:
            logger.error(f"[ERROR] ValueError: {e}")


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
