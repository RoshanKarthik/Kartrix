from langchain.agents.middleware import SummarizationMiddleware

from kartrix.config import settings
from kartrix.llm.factory import get_llm
from kartrix.memory.checkpointer import PgCheckpointSaver
from kartrix.observability.logger import get_logger

logger = get_logger(__name__)


def get_checkpointer() -> PgCheckpointSaver:
    """Postgres-backed LangGraph checkpointer (must be created inside the running event loop)."""
    logger.info("Using Postgres checkpointer")
    return PgCheckpointSaver()


def get_summarization_middleware() -> SummarizationMiddleware:
    return SummarizationMiddleware(
        model=get_llm(),
        trigger=("tokens", settings.memory.summarize_at_tokens),
        keep=("messages", settings.memory.keep_last_messages),
    )
