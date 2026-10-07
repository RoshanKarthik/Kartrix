from langchain.tools import tool

from kartrix.context.index_status import search_note
from kartrix.context.retrievers.pg_hybrid import retrieve
from kartrix.observability.logger import get_logger

logger = get_logger(__name__)


@tool
async def search_codebase(query: str) -> str:
    """
    Search the codebase for relevant classes, functions or logic.
    Use this tool whenever you need to find code related to a question.
    """
    logger.info(f"Tool called: search_codebase with query: {query}")
    chunks = await retrieve(query)
    note = search_note()  # the index may still be building in the background
    if not chunks:
        return "No relevant code found." + (f"\n{note}" if note else "")

    results = []
    for chunk in chunks:
        results.append(
            f"File: {chunk['source']} (lines {chunk['start_line']}-{chunk['end_line']})\n"
            f"Type: {chunk['type']} — {chunk['name']}\n"
            f"Code:\n{chunk['content']}\n"
        )
    if note:
        results.append(note)
    return "\n---\n".join(results)
