"""Plugin base for data sources.

To add a new source (another price feed, a paid fundamentals API, a new
filing type, whatever): drop a new module in src/data/, subclass
`DataSource`, and decorate the class with `@register_source("name")`.
`src/data/__init__.py` auto-imports every module in this package on startup,
so the registration side effect runs with zero changes anywhere else —
`run_pipeline.py` and the scheduler job just iterate `iter_sources()`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

_REGISTRY: dict[str, type["DataSource"]] = {}


def register_source(name: str):
    def _wrap(cls: type["DataSource"]):
        _REGISTRY[name] = cls
        return cls

    return _wrap


def iter_sources() -> dict[str, type["DataSource"]]:
    return dict(_REGISTRY)


class DataSource(ABC):
    """One ingestible data source: a table plus the logic to fetch fresh rows."""

    #: DuckDB table this source's rows get upserted into.
    table: str = ""
    #: Column(s) that uniquely identify a row, used for upsert dedupe.
    key_cols: tuple[str, ...] = ("ticker", "date")

    @abstractmethod
    def fetch(self, tickers: list[str], settings: dict, env) -> pd.DataFrame:
        """Return a DataFrame of fresh rows ready to upsert into `self.table`.

        Implementations should be resilient to a single ticker failing (log
        and skip) rather than aborting the whole batch.
        """
        raise NotImplementedError
