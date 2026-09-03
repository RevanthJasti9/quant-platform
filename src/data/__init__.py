"""Auto-imports every sibling module so their @register_source decorators
run. Adding a new data source module to this directory is enough for it to
be picked up — nothing else in the codebase needs to change.
"""
import importlib
import pkgutil

for _mod in pkgutil.iter_modules(__path__):
    if _mod.name not in ("base", "db", "ingest"):
        importlib.import_module(f"{__name__}.{_mod.name}")
