from datetime import datetime

import pandas as pd
import numpy as np
import pytest

from app.utils.helper import (
    AMSTL,
    BinanceVolumeAnalyzer, 
    TREND_DOWN,
    TREND_SIDEWAYS,
    TREND_UP,
    XGBoostSupportResistancePredictor, 
    _numba_state_machine,
    format_message_events,
    validate_label_quality,
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


# ── AMSTL / _numba_state_machine tests ──────────────────────────────────────

def test_amstl_state_machine_enters_uptrend_after_confirmation():
    """With confirm_bars=3, need 3 consecutive bars above threshold to enter UP."""
    # Bars: 0=sideways, 1-4=above threshold (enters UP after bar 3)
    grad = np.array([0.0, 1.2, 1.3, 1.1, 1.5, 1.4], dtype=np.float64)
    threshold = np.ones_like(grad)

    trends = _numba_state_machine(grad, threshold, confirm_bars=3)

    # Bars 0: sideways (no signal)
    assert trends[0] == TREND_SIDEWAYS
    # Bars 1-2: above threshold but still confirming (count=1, count=2)
    assert trends[1] == TREND_SIDEWAYS
    assert trends[2] == TREND_SIDEWAYS
    # Bar 3: 3rd confirmation bar → transition to UP
    assert trends[3] == TREND_UP
    # Bar 4-5: stays UP
    assert trends[4] == TREND_UP
    assert trends[5] == TREND_UP


def test_amstl_state_machine_single_bar_does_not_kill_trend():
    """A single bar below exit threshold should NOT kill a confirmed trend."""
    # First build a confirmed UP trend, then drop once, then resume
    grad = np.array([1.5, 1.5, 1.5, 0.1, 1.5, 1.5], dtype=np.float64)
    threshold = np.ones_like(grad)

    trends = _numba_state_machine(grad, threshold, confirm_bars=3)

    # Bars 0-2: confirming → enters UP at bar 2
    assert trends[2] == TREND_UP
    # Bar 3: single bar below exit (0.4), but only 1 confirmation → stays UP
    assert trends[3] == TREND_UP
    # Bars 4-5: back above → resets pending, stays UP
    assert trends[4] == TREND_UP
    assert trends[5] == TREND_UP


def test_amstl_state_machine_with_confirm_bars_1_behaves_immediately():
    """With confirm_bars=1, transitions happen immediately (like old behavior)."""
    grad = np.array([0.0, 0.5, 1.2, 1.1], dtype=np.float64)
    threshold = np.ones_like(grad)

    trends = _numba_state_machine(grad, threshold, confirm_bars=1)

    assert trends[0] == TREND_SIDEWAYS
    assert trends[1] == TREND_SIDEWAYS
    assert trends[2] == TREND_UP
    assert trends[3] == TREND_UP


def test_amstl_short_sideways_gap_is_merged_back_into_trend():
    labeler = AMSTL(min_trend_duration=3)
    raw_trend = np.array([TREND_UP, TREND_UP, TREND_UP, TREND_SIDEWAYS, TREND_SIDEWAYS, TREND_UP, TREND_UP, TREND_UP], dtype=np.int8)

    cleaned = labeler._apply_min_duration(raw_trend)

    assert cleaned.tolist() == [TREND_UP] * len(raw_trend)


def test_amstl_short_opposite_burst_is_removed():
    labeler = AMSTL(min_trend_duration=3)
    raw_trend = np.array([TREND_DOWN, TREND_DOWN, TREND_DOWN, TREND_UP, TREND_UP, TREND_DOWN, TREND_DOWN, TREND_DOWN], dtype=np.int8)

    cleaned = labeler._apply_min_duration(raw_trend)

    assert cleaned.tolist() == [
        TREND_DOWN,
        TREND_DOWN,
        TREND_DOWN,
        TREND_SIDEWAYS,
        TREND_SIDEWAYS,
        TREND_DOWN,
        TREND_DOWN,
        TREND_DOWN,
    ]


# ── validate_label_quality tests ────────────────────────────────────────────

def test_validate_label_quality_returns_expected_structure():
    """Basic smoke test for the diagnostic function."""
    # Create a simple trending dataset where UP labels have positive returns
    n = 50
    prices = np.cumsum(np.random.randn(n) * 0.5 + 0.1) + 100  # upward drift
    trends = np.array([TREND_UP] * 20 + [TREND_SIDEWAYS] * 10 + [TREND_DOWN] * 20, dtype=np.int8)
    
    df = pd.DataFrame({"close": prices, "trend": trends})
    result = validate_label_quality(df, forward_bars=5)
    
    assert "up_precision" in result
    assert "down_precision" in result
    assert "sideways_avg_abs_return" in result
    assert "overall_quality" in result
    assert "distribution" in result
    assert result["up_count"] > 0
    assert result["down_count"] > 0
    assert result["sideways_count"] > 0


def test_validate_label_quality_raises_on_missing_columns():
    df = pd.DataFrame({"price": [1, 2, 3]})
    with pytest.raises(ValueError, match="must contain"):
        validate_label_quality(df)


# ── XGBoost integration test ───────────────────────────────────────────────

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
