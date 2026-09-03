"""Central settings: config/settings.yaml (tunable knobs) + .env (secrets)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "settings.yaml"
DB_PATH = ROOT_DIR / "data" / "market.duckdb"
MODELS_DIR = ROOT_DIR / "models"

load_dotenv(ROOT_DIR / ".env")


class EnvSettings(BaseSettings):
    sec_edgar_user_agent: str = "quant-platform-dev unknown@example.com"
    finnhub_api_key: str = ""
    polygon_api_key: str = ""
    groq_api_key: str = ""

    model_config = {"env_file": str(ROOT_DIR / ".env"), "extra": "ignore"}


@lru_cache
def get_env() -> EnvSettings:
    return EnvSettings()


@lru_cache
def get_settings() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)
