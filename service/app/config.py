"""Application configuration: read from environment variables / the .env file.

The .env path is always resolved relative to *this file* to service/.env (the same
layout locally and remotely at /opt/jiayi/service/.env), never relative to the
process working directory — starting uvicorn from the project root or from
service/ behaves identically.
Environment variables always take precedence over the .env file (the
pydantic-settings default; the tests rely on this).
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# The service/ directory (one level above app/)
_SERVICE_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Global service configuration; field names map one-to-one onto the
    environment variables in .env (case-insensitive)."""

    # DeepSeek configuration
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    deepseek_timeout: float = 60.0  # DeepSeek request timeout (seconds)

    # ML model configuration: when empty, or the file does not exist, the
    # built-in heuristic mock model is used instead.
    ml_model_path: str = ""

    # Evaluation data feedback loop: after every completed assessment, append a
    # de-identified record to this JSONL file.
    # (Relative paths resolve against the project root; leave empty to disable.)
    eval_log_path: str = "data/assessments.jsonl"

    # Demo mode: when true, both DeepSeek and the ML model return plausible fake data
    mock_mode: bool = True

    # ---- Web search (the web_search server-side tool of DeepSeek's Anthropic protocol) ----
    #
    # Search is the **only metered, cost-unpredictable** thing in this service: in
    # a measured run, one ordinary question triggered 4 searches and about 13.9k
    # input tokens. That is why all three knobs stay in the configuration — when
    # something goes wrong it can be turned off without a code change.
    #
    # search_enabled            master switch: when false no code path issues a search
    # search_max_uses           per-request cap on search count (passed to the web_search tool)
    # search_cache_ttl_seconds  cache lifetime for destination lookups (keyed by place × language)
    search_enabled: bool = True
    search_max_uses: int = 2
    search_cache_ttl_seconds: int = 6 * 60 * 60

    # Server bind address
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = SettingsConfigDict(
        env_file=str(_SERVICE_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Process-wide singleton configuration (cached by lru_cache; tests can reset
    it with get_settings.cache_clear())."""
    return Settings()
