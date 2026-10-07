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
    # The connection URL (with password) comes from the REDIS_URL env var, never from here.
    enabled: bool = True
    namespace: str = Field("kartrix", pattern=r"^[a-z0-9_-]{1,32}$")  # prefix for every key + index name
    threshold: float = Field(0.85, ge=0.0, le=1.0)
    ttl: int = Field(86400, gt=0)


LLMProvider = Literal["nvidia", "huggingface", "openai", "anthropic"]


class LLMModelRef(_Section):
    provider: LLMProvider
    model: str


class LLMSettings(_Section):
    provider: LLMProvider = "nvidia"
    model: str = "nvidia/nemotron-3-super-120b-a12b"
    judge_model: str | None = None  # cheaper model for LLM-as-judge; falls back to `model`
    router_model: str | None = None  # cheap model that routes each request; falls back to the judge model
    timeout: float = Field(90.0, gt=0)  # per request; free-tier models can hang
    retry: RetrySettings = RetrySettings()
    # Tried in order when the primary still fails after its retries; [] disables fallback.
    fallbacks: list[LLMModelRef] = Field(
        default_factory=lambda: [
            LLMModelRef(provider="nvidia", model="google/gemma-4-31b-it"),
            LLMModelRef(provider="huggingface", model="Qwen/Qwen3-Coder-30B-A3B-Instruct"),
        ]
    )

    @property
    def effective_judge_model(self) -> str:
        return self.judge_model or self.model

    @property
    def effective_router_model(self) -> str:
        return self.router_model or self.effective_judge_model


class SkillsSettings(_Section):
    skills_dir: str = ".kartrix/skills"


class MemorySettings(_Section):
    session_file: str = ".kartrix/current_session"  # id of the active session (thread)
    summarize_at_tokens: int = Field(4000, gt=0)
    keep_last_messages: int = Field(20, gt=0)
    # long-term memory (kartrix.memory.long_term)
    recall_k: int = Field(5, gt=0)
    min_similarity: float = Field(0.3, ge=0.0, le=1.0)
    dedupe_similarity: float = Field(0.92, ge=0.0, le=1.0)  # a new memory this close replaces the old one


class IndexSettings(_Section):
    max_file_kb: int = Field(512, gt=0)  # bigger files are skipped (generated/minified code)
    embed_batch_size: int = Field(32, gt=0)  # chunks per embeddings request
    # gitignore-style patterns always skipped, on top of the repo's own .gitignore files.
    exclude: list[str] = Field(
        default_factory=lambda: [
            ".git/",
            ".kartrix/",
            ".venv/",
            "venv/",
            "node_modules/",
            "__pycache__/",
            ".env",
            ".env.*",
            "*.pem",
            "*.key",
            "*.p12",
            "*.pfx",
            "id_rsa*",
            "id_ed25519*",
        ]
    )


class WorkspaceSettings(_Section):
    """Workspace jail (B1): gitignore-style patterns, relative to the repo root, that file
    tools may not read / write even though they are inside the workspace."""

    deny_read: list[str] = Field(
        default_factory=lambda: [
            ".git/",  # .git/config can hold credentials in remote URLs
            ".kartrix/*",
            "!.kartrix/skills/",  # skill support files are read with read_file
            ".env",
            ".env.*",
            "!.env.example",
            "!.env.sample",
            "!.env.template",
            "*.pem",
            "*.key",
            "*.p12",
            "*.pfx",
            "id_rsa*",
            "id_ed25519*",
            "!*.pub",
        ]
    )
    deny_write: list[str] = Field(
        default_factory=lambda: [
            ".git/",  # writing hooks would run arbitrary code
            ".kartrix/",
            ".env",
            ".env.*",
            "!.env.example",
            "!.env.sample",
            "!.env.template",
            "*.pem",
            "*.key",
            "*.p12",
            "*.pfx",
            "id_rsa*",
            "id_ed25519*",
        ]
    )
    max_file_kb: int = Field(2048, gt=0)  # largest file read_file/edit_file load, or write_file writes


class PermissionsSettings(_Section):
    """Command policy (B2) and permission mode (B4) — see kartrix/security/permissions.py."""

    mode: Literal["read_only", "default", "auto"] = "default"
    # Extra rules, matched against the parsed command: "npm run *" (trailing * = any args),
    # "make test". deny > ask > allow; they never override the built-in hard deny list.
    allow: list[str] = Field(default_factory=list)
    ask: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)
    # Package installs may only use these hosts (and their subdomains) as index/registry.
    registries: list[str] = Field(
        default_factory=lambda: ["pypi.org", "files.pythonhosted.org", "registry.npmjs.org", "registry.yarnpkg.com"]
    )
    command_timeout: float = Field(300.0, gt=0)  # seconds; the whole process tree is killed after this
    # Env vars passed to commands although their name looks secret (e.g. SSH_AUTH_SOCK for git over ssh).
    env_passthrough: list[str] = Field(default_factory=list)


class BudgetLimits(_Section):
    """Limits for one run; null = no limit. Checked before every model and tool call."""

    max_tokens: int | None = Field(None, gt=0)  # input + output tokens of every model call
    max_cost_usd: float | None = Field(None, gt=0)  # needs a price for the model (budgets.prices)
    max_tool_calls: int | None = Field(None, gt=0)
    max_seconds: float | None = Field(None, gt=0)  # wall-clock, without time spent waiting for the user


class ModelPrice(_Section):
    input: float = Field(ge=0)  # USD per million input tokens
    output: float = Field(ge=0)  # USD per million output tokens


class BudgetSettings(_Section):
    """Budgets (B9) — see kartrix/security/budget.py."""

    # One chat request.
    turn: BudgetLimits = BudgetLimits(max_tokens=1_500_000, max_cost_usd=2.0, max_tool_calls=80, max_seconds=1200)
    # One /plan run: planning, every task and the judge.
    plan: BudgetLimits = BudgetLimits(max_tokens=10_000_000, max_cost_usd=10.0, max_tool_calls=600, max_seconds=7200)
    # Model name → price. Models without a price count tokens but no cost (shown as "cost unknown").
    prices: dict[str, ModelPrice] = Field(default_factory=dict)


class CheckpointSettings(_Section):
    """Workspace checkpoints before every turn/task and /undo (B11) — see kartrix/security/checkpoints.py."""

    enabled: bool = True
    keep: int = Field(50, gt=0)  # undo entries kept per workspace
    # gitignore syntax, on top of the workspace's .gitignore files and workspace.deny_write (secrets).
    exclude: list[str] = Field(
        default_factory=lambda: [
            "node_modules/",
            ".venv/",
            "venv/",
            "__pycache__/",
            ".mypy_cache/",
            ".pytest_cache/",
            ".ruff_cache/",
            ".next/",
            ".turbo/",
        ]
    )


class SandboxLimits(_Section):
    memory_mb: int | None = Field(4096, gt=0)  # whole command tree (Windows, Docker); per process elsewhere
    max_processes: int | None = Field(512, gt=0)  # processes the command may run at once (not on macOS)
    max_file_mb: int | None = Field(2048, gt=0)  # largest file a command may write (macOS / Linux)


class DockerSandboxSettings(_Section):
    image: str | None = None  # needed for backend: docker — an image with the project's toolchain
    cpus: float | None = Field(None, gt=0)


class SandboxSettings(_Section):
    """Sandboxed command execution (B8) — see kartrix/sandbox/."""

    # auto: the OS-native sandbox (macOS Seatbelt, Linux bubblewrap or Landlock, Windows AppContainer);
    # native: the same, but an error if it isn't available; docker: always Docker; none: no sandbox
    # (every command that runs code then needs approval).
    backend: Literal["auto", "native", "docker", "none"] = "auto"
    limits: SandboxLimits = SandboxLimits()
    # Paths under the home directory that sandboxed commands can never read (credentials, keys).
    deny_read_home: list[str] = Field(
        default_factory=lambda: [
            ".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker", ".config/gcloud", ".config/gh", ".config/git",
            ".git-credentials", ".netrc", ".npmrc", ".yarnrc", ".yarnrc.yml", ".pypirc", ".password-store",
            ".local/share/keyrings", "Library/Keychains", ".config/kartrix", ".local/share/kartrix",
            "Library/Application Support/kartrix", "AppData/Local/kartrix",
        ]
    )  # fmt: skip
    # Extra folders sandboxed commands may read (e.g. a toolchain outside the usual places; Windows
    # only needs this — elsewhere everything but deny_read_home is readable) or write.
    extra_read: list[str] = Field(default_factory=list)
    extra_write: list[str] = Field(default_factory=list)
    docker: DockerSandboxSettings = DockerSandboxSettings()


class RetrievalSettings(_Section):
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid"  # hybrid = pgvector + full-text, fused with RRF
    top_k: int = Field(5, gt=0)
    candidates: int = Field(40, gt=0)  # results taken from each retriever before fusion
    rrf_k: int = Field(60, gt=0)  # reciprocal-rank-fusion constant
    graph_neighbors: int = Field(2, ge=0)  # callers/callees of the top hits added to search results (0 = off)


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
                    f"Duplicate key {key!r} at line {key_node.start_mark.line + 1} of {key_node.start_mark.name}"
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


class ContextSettings(_Section):
    """Per-turn context for the subagents (kartrix.agent.context): a token budget per labelled section."""

    instructions_tokens: int = Field(1500, gt=0)  # KARTRIX.md (head kept)
    memory_tokens: int = Field(600, gt=0)  # recalled memories, best first, whole items only
    conversation_tokens: int = Field(1500, gt=0)  # recent turns (newest kept)
    repo_map_tokens: int = Field(400, ge=0)  # most-referenced symbols from the code graph (0 = off)


class AgentsSettings(_Section):
    """The multi-agent chat graph (kartrix.agent.graph)."""

    # multi: router + explorer/coder/reviewer subagents (kartrix.agent.graph); single: one ReAct loop
    architecture: Literal["multi", "single"] = "multi"
    max_review_rounds: int = Field(2, ge=1, le=5)  # coder → reviewer rounds before answering anyway


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KARTRIX_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    embeddings: EmbeddingsSettings = EmbeddingsSettings()
    semantic_cache: SemanticCacheSettings = SemanticCacheSettings()
    llm: LLMSettings = LLMSettings()
    agents: AgentsSettings = AgentsSettings()
    context: ContextSettings = ContextSettings()
    skills: SkillsSettings = SkillsSettings()
    memory: MemorySettings = MemorySettings()
    index: IndexSettings = IndexSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    workspace: WorkspaceSettings = WorkspaceSettings()
    permissions: PermissionsSettings = PermissionsSettings()
    budgets: BudgetSettings = BudgetSettings()
    checkpoints: CheckpointSettings = CheckpointSettings()
    sandbox: SandboxSettings = SandboxSettings()
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
