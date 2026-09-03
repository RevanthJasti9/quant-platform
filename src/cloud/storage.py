from __future__ import annotations

from collections import defaultdict

from src.cloud.contracts import ObjectStorage, StructuredStorage


class InMemoryStructuredStorage(StructuredStorage):
    def __init__(self) -> None:
        self._tables: dict[str, list[dict]] = defaultdict(list)

    def write_record(self, table: str, record: dict) -> None:
        self._tables[table].append(dict(record))

    def read_records(self, table: str) -> list[dict]:
        return [dict(row) for row in self._tables.get(table, [])]


class InMemoryObjectStorage(ObjectStorage):
    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    def put_object(self, key: str, payload: bytes, metadata: dict | None = None) -> str:
        self._objects[key] = payload
        return f"memory://{key}"

    def get_object(self, key: str) -> bytes:
        return self._objects[key]
