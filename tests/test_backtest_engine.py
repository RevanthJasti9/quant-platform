import pytest

from src.backtesting import engine as engine_module
from src.backtesting.engine import _build_folds, run_backtest


def test_folds_respect_purge_gap_and_chronology():
    dates = list(range(1500))  # stand-ins for trading-day positions
    settings = {
        "backtest": {"train_window_years": 2, "test_window_months": 6, "step_months": 6},
        "models": {"purge_gap_days": 20},
    }
    folds = _build_folds(dates, settings)
    assert len(folds) > 0
    for train_start, train_end, test_start, test_end in folds:
        assert train_start <= train_end < test_start <= test_end
        assert test_start - train_end - 1 >= settings["models"]["purge_gap_days"]


def test_no_fold_when_history_too_short():
    dates = list(range(100))
    settings = {
        "backtest": {"train_window_years": 3, "test_window_months": 6, "step_months": 6},
        "models": {"purge_gap_days": 20},
    }
    assert _build_folds(dates, settings) == []


def test_run_backtest_rejects_purge_gap_smaller_than_horizon(monkeypatch):
    """A purge gap narrower than the label horizon would let some training
    labels see into the test window -- this must refuse to run rather than
    silently produce an inflated result. Raises before touching the
    database, so no DB fixture is needed here.
    """
    monkeypatch.setattr(engine_module, "get_settings", lambda: {"models": {"purge_gap_days": 5}, "backtest": {}})
    with pytest.raises(RuntimeError, match="purge_gap_days"):
        run_backtest(horizon_days=20)
