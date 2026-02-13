from typing import List, Dict, Any, Optional, Tuple
import re
import json
import time

from .vector_store import VectorStore
from .llm_client import LLMClient
from .database import get_db, Chunk, Document
from sqlalchemy import and_
# Добавьте в HistoryRAGAgent
from .intelligent_search import IntelligentSearch

class HistoryRAGAgent:
    def __init__(self, vector_store: VectorStore, llm_client: LLMClient):
        self.vs = vector_store
        self.llm = llm_client
        self.intelligent_search = IntelligentSearch(vector_store, llm_client)
    
    def answer_fact(self, query: str, document_id: Optional[int] = None, top_k: int = 5) -> Dict[str, Any]:
        """
        Умный ответ с пониманием контекста
        """
        # Используем интеллектуальный поиск
        result = self.intelligent_search.answer_question(query)
        
        # Фильтруем по document_id если нужно
        if document_id and result['sources']:
            result['sources'] = [s for s in result['sources'] 
                                if str(document_id) in str(s.get('doc_id', ''))]
        
        return result