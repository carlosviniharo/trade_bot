from datetime import datetime

import pandas as pd
import pytest

from app.utils.helper import BinanceVolumeAnalyzer


@pytest.fixture()
def analyzer() -> BinanceVolumeAnalyzer:
    return BinanceVolumeAnalyzer()


def _build_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows)


def test_get_top_symbols_returns_empty_when_df_empty(analyzer: BinanceVolumeAnalyzer):
    analyzer.df_final_values = pd.DataFrame()

    result = analyzer.get_top_symbols(metric="volume_change", ascending=False, n_values=3)
    assert result == []


def test_get_top_symbols_returns_empty_when_metric_invalid(analyzer: BinanceVolumeAnalyzer):
    analyzer.df_final_values = _build_df([
        {
            "symbol": "BTCUSDT",
            "event_timestamp": pd.Timestamp(datetime(2024, 1, 1, 0, 0, 0)),
            "price_change": 1.0,
            "volume_change": 10.0,
            "atr_pct": 2.0,
            "close": 42000.0,
        }
    ])

    result = analyzer.get_top_symbols(metric="nonexistent_metric", ascending=False, n_values=3)
    assert result == []


@pytest.mark.parametrize(
    "metric,ascending,n_values,expected_symbols,flag_key",
    [
        ("volume_change", False, 2, ["ETHUSDT", "SOLUSDT"], "is_volume_event"),
        ("price_change", True, 2, ["ETHUSDT", "SOLUSDT"], "is_price_event"),
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
            "price_change": 3.0,
            "volume_change": 5.0,
            "atr_pct": 2.0,
            "close": 42000.0,
        },
        {
            "symbol": "ETHUSDT",
            "event_timestamp": pd.Timestamp(datetime(2024, 1, 1, 0, 0, 0)),
            "price_change": -4.0,
            "volume_change": 15.0,
            "atr_pct": 1.5,
            "close": 3200.0,
        },
        {
            "symbol": "SOLUSDT",
            "event_timestamp": pd.Timestamp(datetime(2024, 1, 1, 0, 0, 0)),
            "price_change": 1.0,
            "volume_change": 9.0,
            "atr_pct": 3.0,
            "close": 110.0,
        },
    ])

    result = analyzer.get_top_symbols(metric=metric, ascending=ascending, n_values=n_values)

    assert [row["symbol"] for row in result] == expected_symbols
    for row in result:
        assert row.get(flag_key) is True
        # Ensure the other flag is not present
        other_flag = "is_price_event" if flag_key == "is_volume_event" else "is_volume_event"
        assert other_flag not in row


