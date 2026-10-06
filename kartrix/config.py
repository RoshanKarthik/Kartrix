"""Typed application settings.

Values come from (highest priority first):
  1. environment variables prefixed ``KARTRIX_`` — nested keys use ``__``,
     e.g. ``KARTRIX_LLM__MODEL=gpt-4o`` overrides ``llm.model``
  2. ``kartrix/config.yaml`` (or the file named by ``KARTRIX_CONFIG_FILE``)
  3. the defaults declared on the models below

The YAML is loaded strictly: duplicate keys and unknown keys are errors, so a
typo or a copy-pasted section can no longer silently override another one.
Secrets never live here — they stay in env vars read by the code that needs them.
"""

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    InitSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_FILE = Path(__file__).parent / "config.yaml"

# Load .env before settings are built so KARTRIX_* overrides placed there apply.
load_dotenv(PROJECT_ROOT / ".env")


class ConfigError(Exception):
    """Raised when config.yaml cannot be loaded or fails validation."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrySettings(_Section):
    """Exponential backoff with jitter for transient errors (timeouts, 429, 5xx)."""

    max_retries: int = Field(2, ge=0, le=10)
    initial_delay: float = Field(1.0, gt=0)
    backoff_factor: float = Field(2.0, ge=1.0)
    max_delay: float = Field(30.0, gt=0)


class EmbeddingsSettings(_Section):
    # No fallback provider on purpose: vectors from different models are not comparable.
    provider: Literal["nvidia", "openai", "huggingface"] = "nvidia"
    model: str = "nvidia/nemotron-3-embed-1b"
    dims: int = Field(2048, gt=0)
    timeout: float = Field(60.0, gt=0)
    retry: RetrySettings = RetrySettings()


class SemanticCacheSettings(_Section):
    enabled: bool = True
    redis_url: str = "redis://localhost:6379"
    threshold: float = Field(0.85, ge=0.0, le=1.0)
    ttl: int = Field(86400, gt=0)


class TasksSettings(_Section):
    db_path: str = ".kartrix/tasks.db"


LLMProvider = Literal["nvidia", "huggingface", "openai", "anthropic"]


class LLMModelRef(_Section):
    provider: LLMProvider
    model: str


class LLMSettings(_Section):
    provider: LLMProvider = "nvidia"
    model: str = "nvidia/nemotron-3-super-120b-a12b"
    judge_model: str | None = None  # cheaper model for LLM-as-judge; falls back to `model`
    timeout: float = Field(90.0, gt=0)  # per request; free-tier models can hang
    retry: RetrySettings = RetrySettings()
    # Tried in order when the primary still fails after its retries; [] disables fallback.
    fallbacks: list[LLMModelRef] = Field(default_factory=lambda: [
        LLMModelRef(provider="nvidia", model="google/gemma-4-31b-it"),
        LLMModelRef(provider="huggingface", model="Qwen/Qwen3-Coder-30B-A3B-Instruct"),
    ])

    @property
    def effective_judge_model(self) -> str:
        return self.judge_model or self.model


class SkillsSettings(_Section):
    skills_dir: str = ".kartrix/skills"


class MemorySettings(_Section):
    db_path: str = ".kartrix/memory/memory.db"
    summarize_at_tokens: int = Field(4000, gt=0)
    keep_last_messages: int = Field(20, gt=0)


class IndexSettings(_Section):
    max_file_kb: int = Field(512, gt=0)          # bigger files are skipped (generated/minified code)
    embed_batch_size: int = Field(32, gt=0)      # chunks per embeddings request
    # gitignore-style patterns always skipped, on top of the repo's own .gitignore files.
    exclude: list[str] = Field(default_factory=lambda: [
        ".git/", ".kartrix/", ".venv/", "venv/", "node_modules/", "__pycache__/",
        ".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "id_rsa*", "id_ed25519*",
    ])


class RetrievalSettings(_Section):
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid"  # hybrid = pgvector + full-text, fused with RRF
    top_k: int = Field(5, gt=0)
    candidates: int = Field(40, gt=0)            # results taken from each retriever before fusion
    rrf_k: int = Field(60, gt=0)                 # reciprocal-rank-fusion constant


class DatabaseSettings(_Section):
    # The connection URL (with password) comes from the DATABASE_URL env var, never from here.
    pool_size: int = Field(5, gt=0)
    max_overflow: int = Field(10, ge=0)
    pool_timeout: float = Field(30.0, gt=0)
    echo: bool = False  # log every SQL statement (debug only)


class LoggingSettings(_Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["json", "text"] = "json"
    # Kept quiet by default so log lines don't interleave with the interactive REPL.
    console_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING"
    file: str | None = ".kartrix/logs/kartrix.jsonl"  # null disables file logging


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of keeping the last one."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise ConfigError(
                    f"Duplicate key {key!r} at line {key_node.start_mark.line + 1} "
                    f"of {key_node.start_mark.name}"
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def load_yaml_strict(path: Path) -> dict[str, Any]:
    """Parse a YAML file, failing on duplicate keys or a non-mapping document."""
    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.load(fh, Loader=_UniqueKeyLoader)  # noqa: S506 — SafeLoader subclass
    except FileNotFoundError as e:
        raise ConfigError(f"Config file not found: {path}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KARTRIX_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    embeddings: EmbeddingsSettings = EmbeddingsSettings()
    semantic_cache: SemanticCacheSettings = SemanticCacheSettings()
    tasks: TasksSettings = TasksSettings()
    llm: LLMSettings = LLMSettings()
    skills: SkillsSettings = SkillsSettings()
    memory: MemorySettings = MemorySettings()
    index: IndexSettings = IndexSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    database: DatabaseSettings = DatabaseSettings()
    logging: LoggingSettings = LoggingSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        config_file = Path(os.environ.get("KARTRIX_CONFIG_FILE", DEFAULT_CONFIG_FILE))
        yaml_settings = InitSettingsSource(settings_cls, init_kwargs=load_yaml_strict(config_file))
        return init_settings, env_settings, yaml_settings


def load_settings() -> Settings:
    """Build and validate settings; raises ConfigError with a readable message."""
    try:
        return Settings()
    except ConfigError:
        raise
    except Exception as e:  # pydantic.ValidationError, bad env values, ...
        raise ConfigError(f"Invalid configuration:\n{e}") from e


settings = load_settings()
