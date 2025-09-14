from datetime import datetime

import pandas as pd
import pytest

from app.utils.helper import BinanceVolumeAnalyzer, format_message_spikes

# Test cases for BaseAnalyzer and BinanceVolumeAnalyzer

@pytest.fixture()
def analyzer() -> BinanceVolumeAnalyzer:
    return BinanceVolumeAnalyzer()


def _build_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows)


def test_get_top_symbols_raises_when_df_empty(analyzer: BinanceVolumeAnalyzer):
    analyzer.df_final_values = pd.DataFrame()
    with pytest.raises(ValueError, match="DataFrame is empty"):
        analyzer.get_top_symbols(metric="volume_rate", ascending=False, n_values=3)


def test_get_top_symbols_raises_when_metric_invalid(analyzer: BinanceVolumeAnalyzer):
    analyzer.df_final_values = _build_df([
        {
            "symbol": "BTCUSDT",
            "event_timestamp": pd.Timestamp(datetime(2024, 1, 1, 0, 0, 0)),
            "price_rate": 1.0,
            "volume_rate": 10.0,
            "atr_pct": 2.0,
            "close": 42000.0,
        }
    ])

    with pytest.raises(ValueError, match="Metric 'nonexistent_metric' not found in DataFrame columns"):
        analyzer.get_top_symbols(metric="nonexistent_metric", ascending=False, n_values=3)


@pytest.mark.parametrize(
    "metric,ascending,n_values,expected_symbols,flag_key",
    [
        ("volume_rate", False, 2, ["ETHUSDT", "SOLUSDT"], "is_volume_event"),
        ("price_rate", True, 2, ["ETHUSDT", "SOLUSDT"], "is_price_event"),
    ],
)
def test_get_top_symbols_flags_and_sorting(
    analyzer: BinanceVolumeAnalyzer,
    metric: str,
    ascending: bool,
    n_values: int,
    expected_symbols: list[str],
    flag_key: str,
):
    analyzer.df_final_values = _build_df([
        {
            "symbol": "BTCUSDT",
            "event_timestamp": pd.Timestamp(datetime(2024, 1, 1, 0, 0, 0)),
            "price_rate": 3.0,
            "volume_rate": 5.0,
            "atr_pct": 2.0,
            "close": 42000.0,
        },
        {
            "symbol": "ETHUSDT",
            "event_timestamp": pd.Timestamp(datetime(2024, 1, 1, 0, 0, 0)),
            "price_rate": -4.0,
            "volume_rate": 15.0,
            "atr_pct": 1.5,
            "close": 3200.0,
        },
        {
            "symbol": "SOLUSDT",
            "event_timestamp": pd.Timestamp(datetime(2024, 1, 1, 0, 0, 0)),
            "price_rate": 1.0,
            "volume_rate": 9.0,
            "atr_pct": 3.0,
            "close": 110.0,
        },
    ])

    result = analyzer.get_top_symbols(metric=metric, ascending=ascending, n_values=n_values)

    assert result.symbol.tolist() == expected_symbols

    for row in result.itertuples(index=True, name='Row'):
        value = getattr(row, flag_key)
        assert value is True
        # Ensure the other flag is not present
        other_flag = "is_price_event" if flag_key == "is_volume_event" else "is_volume_event"
        assert other_flag not in row


# Test cases for format_message_spikes

def test_format_message_spikes_empty():
    assert format_message_spikes() == ""

def test_format_message_spikes_all_filtered():
    # All below threshold
    rows = [
        {
            "symbol": "FOO",
            "price_rate": 1.0,
            "volume_rate": 1000,
            "atr_pct": 0.2,
            "close": 10,
        },
        {
            "symbol": "BAR",
            "price_rate": 2.0,
            "volume_rate": 2000,
            "atr_pct": 0.3,
            "close": 20,
        },
    ]
    assert format_message_spikes(*rows) == ""

def test_format_message_spikes_invalid_values():
    # Should skip rows with invalid float conversion
    rows = [
        {
            "symbol": "BAD",
            "price_rate": "not_a_number",
            "volume_rate": 6000,
            "atr_pct": 1.0,
            "close": 100,
        },
        {
            "symbol": "GOOD",
            "price_rate": 4.0,
            "volume_rate": 6000,
            "atr_pct": 1.0,
            "close": 100,
        },
    ]
    result = format_message_spikes(*rows)
    assert "Symbol: GOOD" in result
    assert "Symbol: BAD" not in result
