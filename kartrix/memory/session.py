import uuid
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert

from kartrix.config import settings
from kartrix.db.engine import session_scope
from kartrix.db.models import Session
from kartrix.observability.logger import get_logger


logger = get_logger(__name__)


class InvalidSessionIdError(ValueError):
   """Raised when a session id is not a canonical UUID4 string."""


def validate_session_id(session_id: str) -> str:
   """Return the session id if it is a canonical, lowercase UUID4 string.

   Session ids become LangGraph thread ids and DB keys, so only the exact form we
   generate is accepted — no braces, ``urn:uuid:`` prefixes, missing hyphens or
   other values that ``uuid.UUID`` would otherwise tolerate.
   """
   candidate = session_id.strip() if isinstance(session_id, str) else ""
   if len(candidate) != 36:
       raise InvalidSessionIdError(f"Invalid session id: {session_id!r}")
   try:
       parsed = uuid.UUID(candidate)
   except ValueError:
       raise InvalidSessionIdError(f"Invalid session id: {session_id!r}") from None
   if parsed.version != 4 or str(parsed) != candidate.lower():
       raise InvalidSessionIdError(f"Invalid session id: {session_id!r}")
   return str(parsed)


def _session_file() -> Path:
   return Path(settings.memory.session_file)


def get_current_session() -> str:
   session_file = _session_file()
   if session_file.exists():
       stored = session_file.read_text().strip()
       try:
           session_id = validate_session_id(stored)
       except InvalidSessionIdError:
           logger.warning("Stored session id is invalid; starting a new session", extra={"stored": stored[:64]})
           return new_session()
       logger.info("Resuming session", extra={"session_id": session_id})
       return session_id
   return new_session()


def new_session() -> str:
   session_id = str(uuid.uuid4())
   session_file = _session_file()
   session_file.parent.mkdir(parents=True, exist_ok=True)
   session_file.write_text(session_id)
   logger.info("Started new session", extra={"session_id": session_id})
   return session_id


def switch_session(session_id: str) -> str:
   """Make ``session_id`` current. Raises InvalidSessionIdError for malformed ids."""
   session_id = validate_session_id(session_id)
   session_file = _session_file()
   session_file.parent.mkdir(parents=True, exist_ok=True)
   session_file.write_text(session_id)
   logger.info("Switched session", extra={"session_id": session_id})
   return session_id


async def record_session(session_id: str, repo_path: str) -> None:
   """Make sure the session has a row in Postgres (projects link to it); touch updated_at."""
   session_id = validate_session_id(session_id)
   stmt = insert(Session).values(id=uuid.UUID(session_id), repo_path=repo_path)
   stmt = stmt.on_conflict_do_update(index_elements=[Session.id], set_={"updated_at": stmt.excluded.updated_at})
   async with session_scope() as s:
       await s.execute(stmt)
