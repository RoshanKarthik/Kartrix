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


class EmbeddingsSettings(_Section):
    provider: Literal["openai", "huggingface"] = "openai"
    model: str = "text-embedding-3-small"
    dims: int = Field(1536, gt=0)


class SemanticCacheSettings(_Section):
    enabled: bool = True
    redis_url: str = "redis://localhost:6379"
    threshold: float = Field(0.85, ge=0.0, le=1.0)
    ttl: int = Field(86400, gt=0)


class TasksSettings(_Section):
    db_path: str = ".kartrix/tasks.db"


class LLMSettings(_Section):
    provider: Literal["openai", "anthropic"] = "openai"
    model: str = "gpt-5.5"
    judge_model: str | None = None  # cheaper model for LLM-as-judge; falls back to `model`

    @property
    def effective_judge_model(self) -> str:
        return self.judge_model or self.model


class SkillsSettings(_Section):
    skills_dir: str = ".kartrix/skills"


class MemorySettings(_Section):
    db_path: str = ".kartrix/memory/memory.db"
    summarize_at_tokens: int = Field(4000, gt=0)
    keep_last_messages: int = Field(20, gt=0)


class ChromaSettings(_Section):
    persist_dir: str = ".kartrix/chromadb/"
    collection_name: str = "codebase"


class RAGSettings(_Section):
    mode: Literal["semantic", "hybrid"] = "semantic"


class VectorStoreSettings(_Section):
    provider: Literal["chroma", "qdrant"] = "chroma"
    retrieval_mode: Literal["dense", "sparse", "hybrid"] = "dense"


class QdrantSettings(_Section):
    collection_name: str = "kartrix"


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
    chromadb: ChromaSettings = ChromaSettings()
    rag: RAGSettings = RAGSettings()
    vector_store: VectorStoreSettings = VectorStoreSettings()
    qdrant: QdrantSettings = QdrantSettings()
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
