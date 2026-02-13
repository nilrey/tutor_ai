# app/fact_retrieval.py
from typing import List, Dict, Any
import re
from sqlalchemy.orm import Session
from sqlalchemy import or_

from .database import Chunk, get_db
from .vector_store import VectorStore


class FactRetrievalEngine:
    """
    Строгий retrieval-движок для фактологических запросов.
    Retrieval-first, без LLM.
    """

    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store

    # ---------- ENTITY EXTRACTION ----------

    def extract_entities(self, query: str) -> List[str]:
        """
        Простейший NER:
        Ищет имена собственные, даты, ключевые слова.
        """
        # Имена собственные (Цезарь, Наполеон, Сталин)
        names = re.findall(r"[А-ЯЁ][а-яё]+", query)

        # Даты
        years = re.findall(r"\b\d{3,4}\b", query)

        # Ключевые слова
        keywords = []
        for word in query.lower().split():
            if len(word) > 4:
                keywords.append(word)

        entities = list(set(names + years + keywords))
        return entities

    # ---------- SQL LEXICAL SEARCH ----------

    def sql_lexical_search(self, db: Session, entities: List[str], limit: int = 50) -> List[Chunk]:
        """
        Жёсткий поиск по SQL (LIKE)
        """
        if not entities:
            return []

        filters = []
        for ent in entities:
            filters.append(Chunk.content.ilike(f"%{ent}%"))

        results = (
            db.query(Chunk)
            .filter(or_(*filters))
            .limit(limit)
            .all()
        )

        return results

    # ---------- SEMANTIC SEARCH ----------

    def semantic_search(self, query: str, n_results: int = 20) -> List[Dict[str, Any]]:
        """
        Поиск по векторному хранилищу
        """
        results = self.vs.search(query, n_results=n_results)

        chunks = []

        if results and results.get("documents"):
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i]
                dist = results["distances"][0][i] if results.get("distances") else 1.0

                chunks.append({
                    "content": doc,
                    "metadata": meta,
                    "distance": dist
                })

        return chunks

    # ---------- HYBRID MERGE ----------

    def merge_results(
        self,
        sql_chunks: List[Chunk],
        semantic_chunks: List[Dict[str, Any]],
        entities: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Пересечение + ранжирование
        """
        merged = []

        # SQL chunks -> dict
        sql_map = {}
        for ch in sql_chunks:
            key = f"{ch.doc_id}_{ch.page_number}_{ch.chunk_index}"
            sql_map[key] = ch

        for sem in semantic_chunks:
            meta = sem["metadata"]
            key = f"{meta.get('doc_id')}_{meta.get('page_number')}_{meta.get('chunk_index')}"

            # Если есть пересечение SQL + semantic → приоритет
            if key in sql_map:
                text = sem["content"]
                score = 0

                # Entity score
                for ent in entities:
                    if ent.lower() in text.lower():
                        score += 2

                # Distance score
                score += max(0, 1 - sem["distance"])

                merged.append({
                    "content": text,
                    "metadata": meta,
                    "score": score,
                    "source": "hybrid"
                })

        # Если пересечений мало — добавляем сильные semantic
        if len(merged) < 3:
            for sem in semantic_chunks:
                text = sem["content"]
                score = max(0, 1 - sem["distance"])
                merged.append({
                    "content": text,
                    "metadata": sem["metadata"],
                    "score": score,
                    "source": "semantic"
                })

        # Сортировка
        merged.sort(key=lambda x: x["score"], reverse=True)

        return merged[:5]

    # ---------- MAIN PIPELINE ----------

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Главный метод retrieval
        """
        db = get_db()
        try:
            entities = self.extract_entities(query)

            sql_chunks = self.sql_lexical_search(db, entities, limit=50)
            semantic_chunks = self.semantic_search(query, n_results=20)

            merged = self.merge_results(sql_chunks, semantic_chunks, entities)

            return merged

        finally:
            db.close()
