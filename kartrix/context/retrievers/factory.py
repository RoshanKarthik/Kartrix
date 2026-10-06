from kartrix.config import settings
from kartrix.observability.logger import get_logger


logger = get_logger(__name__)




def get_retriever():
   mode = settings.rag.mode
   provider = settings.vector_store.provider


   if mode == "hybrid" and provider == "qdrant":
       from .hybrid_qdrant import retrieve
   elif provider == "qdrant":
       from .semantic_qdrant import retrieve
   else:
       from .semantic_chroma import retrieve


   return retrieve
