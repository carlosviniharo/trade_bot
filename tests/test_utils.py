from datetime import datetime

import pandas as pd
import numpy as np
import pytest

from app.utils.helper import (
    BinanceVolumeAnalyzer, 
    XGBoostSupportResistancePredictor, 
    format_message_events
)


# Test cases for BaseAnalyzer and BinanceVolumeAnalyzer

@pytest.fixture()
def analyzer() -> BinanceVolumeAnalyzer:
    return BinanceVolumeAnalyzer()


def _build_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows)


def test_get_top_symbols_raises_when_df_empty(analyzer: BinanceVolumeAnalyzer):
    analyzer._df_final_values = pd.DataFrame()
    with pytest.raises(ValueError, match="DataFrame is empty"):
        analyzer.get_top_symbols(metric="volume_rate", ascending=False, n_values=3)


def test_get_top_symbols_raises_when_metric_invalid(analyzer: BinanceVolumeAnalyzer):
    analyzer._df_final_values = _build_df([
        {
            "symbol": "BTCUSDT",
            "event_timestamp": pd.Timestamp(datetime(2024, 1, 1, 0, 0, 0)),
            "price_rate": 1.0,
            "atr_pct": 2.0,
            "close": 42000.0,
        }
    ])

    with pytest.raises(ValueError, match="Unsupported metric 'nonexistent_metric'. Expected 'price_rate'."):
        analyzer.get_top_symbols(metric="nonexistent_metric", ascending=False, n_values=3)


@pytest.mark.parametrize(
    "metric,ascending,n_values,expected_symbols,threshold",
    [
        ("price_rate", False, 2, ["SOLUSDT", "BTCUSDT"], 2),
    ],
)
def test_get_top_symbols_and_sorting(
    analyzer: BinanceVolumeAnalyzer,
    metric: str,
    ascending: bool,
    n_values: int,
    expected_symbols: list[str],
    threshold: int,
):
    analyzer._df_final_values = _build_df([
        {
            "symbol": "BTCUSDT",
            "event_timestamp": pd.Timestamp(datetime(2024, 1, 1, 0, 0, 0)),
            "price_rate": 3.0,
            "atr_pct": 2.0,
            "close": 42000.0,
        },
        {
            "symbol": "ETHUSDT",
            "event_timestamp": pd.Timestamp(datetime(2024, 1, 1, 0, 0, 0)),
            "price_rate": -4.0,
            "atr_pct": 1.5,
            "close": 3200.0,
        },
        {
            "symbol": "SOLUSDT",
            "event_timestamp": pd.Timestamp(datetime(2024, 1, 1, 0, 0, 0)),
            "price_rate": 5.0,
            "atr_pct": 3.0,
            "close": 110.0,
        },
    ])

    result = analyzer.get_top_symbols(metric=metric, ascending=ascending, n_values=n_values, threshold=threshold)
    print(result)

    assert result.symbol.tolist() == expected_symbols

# Test cases for format_message_events

def test_format_message_events_empty():
    assert format_message_events() == ""

def test_format_message_events_no_filtering():
    # Should format all messages regardless of threshold
    rows = [
        {
            "symbol": "FOO",
            "price_rate": 1.0,
            "atr_pct": 0.2,
            "close": 10,
        },
        {
            "symbol": "BAR",
            "price_rate": 2.0,
            "atr_pct": 0.3,
            "close": 20,
        },
    ]
    expected_output = (
        "\nSymbol: FOO\n"
        "Price Change: 1.00%\n"
        "ATR Percentage: 0.20%\n"
        "Close Price: 10\n"
        "──────────────\n"
        "\nSymbol: BAR\n"
        "Price Change: 2.00%\n"
        "ATR Percentage: 0.30%\n"
        "Close Price: 20\n"
        "──────────────"
    )
    assert format_message_events(*rows) == expected_output


def test_format_message_events_invalid_values(caplog):
    # Should catch ValueError and return empty string (logs error instead of crashing)
    rows = [
        {
            "symbol": "BAD",
            "price_rate": "not_a_number",
            "atr_pct": 1.0,
            "close": 100,
        },
    ]

    # No exception raised, returns empty string for failed items
    result = format_message_events(*rows)
    assert result == ""
    
    # Verify that ValueError was caught and logged
    assert len(caplog.records) > 0
    assert any("ValueError" in record.message for record in caplog.records)




# Mock exchange to avoid network calls
class MockExchange:
    async def fetch_ohlcv(self, symbol, timeframe, limit):
        # Generate dummy OHLCV data
        dates = pd.date_range(end=pd.Timestamp.now(), periods=limit, freq='15min')
        data = []
        for d in dates:
            data.append([
                d.timestamp() * 1000,
                100.0 + np.random.randn(),
                105.0 + np.random.randn(),
                95.0 + np.random.randn(),
                102.0 + np.random.randn(),
                1000.0 + np.random.randn()
            ])
        return data

    async def close(self):
        pass

@pytest.mark.asyncio
async def test_xgboost_predictor_full_flow():
    # Disable hyperparam tuning for speed in verification
    predictor = XGBoostSupportResistancePredictor(tune_hyperparams=False, n_splits=3)
    
    # Inject mock exchange
    predictor.exchange = MockExchange()
    
    # 1. Test get_historical_data
    df = await predictor.get_historical_data("BTC/USDT", limit=500)
    assert not df.empty, "DataFrame should not be empty"
    assert "close" in df.columns, "DataFrame should have 'close' column"
    assert isinstance(df.index, pd.DatetimeIndex) or df.index.name == "timestamp", "Index should be timestamp"
    
    # 2. Test train
    await predictor.train(df)
    assert predictor.model_high is not None, "Model high should be trained"
    assert predictor.model_low is not None, "Model low should be trained"
    
    # 3. Test predict_latest
    prediction = await predictor.predict_latest(lookback=50) # Use smaller lookback for small mock data
    assert prediction is not None, "Prediction should not be None"
    assert "resistance" in prediction, "Prediction should contain 'resistance'"
    assert "support" in prediction, "Prediction should contain 'support'"
    assert prediction["resistance"] != 0, "Resistance should be non-zero"

