from typing import List, Dict, Any, Optional, Tuple
import re
import json
import time

from .vector_store import VectorStore
from .llm_client import LLMClient
from .database import get_db, Chunk, Document
from sqlalchemy import and_

class HistoryRAGAgent:
    def __init__(self, vector_store: VectorStore, llm_client: LLMClient):
        self.vs = vector_store
        self.llm = llm_client
        
#         # Адаптированные промпты для локальных моделей
#         self.fact_system_prompt = """Ты — профессор истории. Отвечай ТОЛЬКО по тексту учебника.

# Правила:
# 1. Если ответа нет в тексте - скажи "Ответ не найден в учебнике"
# 2. В конце ответа ОБЯЗАТЕЛЬНО укажи: Источник: [Глава, §, стр.]
# 3. Не придумывай факты

# Будь краток и точен."""

        self.fact_system_prompt = """Ты — профессор истории. Отвечай ТОЛЬКО по тексту учебника.

        ВАЖНО: 
        1. Если в тексте есть прямое упоминание того, о чем спрашивают - используй ЭТОТ фрагмент
        2. Не используй информацию о других исторических личностях, если вопрос про конкретного человека
        3. Если не находишь точного ответа - скажи "Ответ не найден в учебнике"

        В конце ответа обязательно укажи: Источник: [стр. X, Глава Y, §Z]"""
        
        # Для генерации вопросов - более простой промпт
        self.questions_system_prompt = """Составь вопросы по тексту. 
Формат:
1. Вопрос
2. Вопрос

Ответы:
1. Ответ [стр. X]
2. Ответ [стр. X]"""

def answer_fact(self, query: str, document_id: Optional[int] = None, top_k: int = 5) -> Dict[str, Any]:
    start_time = time.time()
    
    # Используем гибридный поиск
    combined_results = self.vs.hybrid_search(query, n_results=top_k)
    
    if not combined_results:
        return {
            "answer": "❌ Не удалось найти информацию в учебнике.",
            "sources": [],
            "processing_time": time.time() - start_time
        }
    
    # Фильтрация по document_id если указан
    if document_id:
        combined_results = [r for r in combined_results 
                          if int(r['metadata'].get('doc_id', 0)) == document_id]
    
    # Сборка контекста
    context_parts = []
    sources = []
    
    for i, result in enumerate(combined_results):
        meta = result['metadata']
        context_parts.append(f"[Фрагмент {i+1} | Стр. {meta.get('page_number', '?')} | {meta.get('chapter', '')} {meta.get('paragraph', '')}]")
        context_parts.append(result['content'][:800])  # Ограничиваем
        context_parts.append("---")
        
        sources.append({
            "page": meta.get('page_number', '?'),
            "chapter": meta.get('chapter', ''),
            "paragraph": meta.get('paragraph', ''),
            "source_type": result.get('source', 'vector')
        })
    
    context = "\n".join(context_parts)
    
    # Формируем промпт с акцентом на точное совпадение
    prompt = f"""Текст учебника:
{context}

Вопрос: {query}

Найди в тексте информацию, ОТНОСЯЩУЮСЯ К ВОПРОСУ.
Особое внимание обрати на фрагменты, где есть слова: {query}

Ответь кратко и укажи источник:"""
    
    raw_answer = self.llm.generate(
        prompt=prompt,
        system_message=self.fact_system_prompt,
        temperature=0.0
    )
    
    return {
        "answer": raw_answer,
        "sources": sources,
        "processing_time": time.time() - start_time
    }


#     def answer_fact(self, query: str, document_id: Optional[int] = None, top_k: int = 2) -> Dict[str, Any]:
#         """
#         Режим 1: Ответ на фактологический вопрос
#         """
#         start_time = time.time()
#         top_k = min(top_k, 2) # Не более 2 чанков для Ollama
        
#         # 1. Поиск релевантных чанков
#         search_results = self.vs.search(query, n_results=top_k)
        
#         if not search_results or not search_results.get('documents'):
#             return {
#                 "answer": "❌ Не удалось найти информацию в учебнике.",
#                 "sources": [],
#                 "processing_time": time.time() - start_time
#             }
        
#         # 2. Фильтрация по document_id если указан
#         documents = search_results['documents'][0]
#         metadatas = search_results['metadatas'][0]
#         distances = search_results.get('distances', [[]])[0]
        
#         if document_id:
#             filtered = []
#             for i, meta in enumerate(metadatas):
#                 if int(meta.get('doc_id', 0)) == document_id:
#                     filtered.append((documents[i], meta, distances[i] if i < len(distances) else None))
#             if not filtered:
#                 return {
#                     "answer": f"❌ Документ с ID {document_id} не содержит информации по вашему запросу.",
#                     "sources": [],
#                     "processing_time": time.time() - start_time
#                 }
#             documents = [f[0] for f in filtered]
#             metadatas = [f[1] for f in filtered]
#             distances = [f[2] for f in filtered if f[2] is not None]
        
#         # 3. Сборка контекста
#         context_parts = []
#         for i, (doc, meta) in enumerate(zip(documents, metadatas)):
#             context_parts.append(f"[Фрагмент {i+1} | Стр. {meta.get('page_number', '?')} | Глава: {meta.get('chapter', '?')} | §: {meta.get('paragraph', '?')}]")
#             context_parts.append(doc[:1500])  # Ограничиваем длину
#             context_parts.append("---")
        
#         context = "\n".join(context_parts)
        
#         # 4. Формируем промпт
#         prompt = f"""КОНТЕКСТ (ТОЛЬКО ЭТОТ ТЕКСТ ИСПОЛЬЗУЙ ДЛЯ ОТВЕТА):
# {context}

# ВОПРОС ПОЛЬЗОВАТЕЛЯ: {query}

# Твой ответ (строго по контексту с указанием источников):"""
        
#         # 5. Отправляем в LLM
#         raw_answer = self.llm.generate(
#             prompt=prompt,
#             system_message=self.fact_system_prompt,
#             temperature=0.0
#         )
        
#         # 6. Парсим ответ и ссылки
#         answer, sources = self._parse_fact_answer(raw_answer)
        
#         # 7. Валидация источников
#         valid_sources = self._validate_sources(sources, metadatas)
        
#         # 8. СОХРАНЯЕМ В БД
#         try:
#             self.save_qa_to_db(
#                 query=query,
#                 answer=answer,
#                 sources=valid_sources,
#                 mode="fact",
#                 topic=query[:100]  # Первые 100 символов как тема
#             )
#         except Exception as e:
#             print(f"⚠️ Ошибка сохранения в БД: {e}")

#         return {
#             "answer": answer,
#             "sources": valid_sources,
#             "raw_sources": metadatas,  # Для отладки
#             "confidence": 1.0 - (distances[0] if distances else 0),
#             "processing_time": time.time() - start_time
#         }
    


    def generate_questions(self, document_id: int, chapter: str, paragraph: str, num_questions: int = 5) -> Dict[str, Any]:
        """
        Режим 2: Генерация вопросов по конкретному параграфу
        """
        start_time = time.time()
        
        # 1. Получаем чанки из SQL по точным метаданным
        db = get_db()
        try:
            # Ищем чанки, содержащие указанный параграф
            chunks = db.query(Chunk).filter(
                Chunk.doc_id == document_id,
                Chunk.paragraph.contains(paragraph) if paragraph else True
            ).order_by(Chunk.page_number).all()
            
            if not chunks:
                # Пробуем поиск по тексту
                chunks = db.query(Chunk).filter(
                    Chunk.doc_id == document_id,
                    Chunk.content.contains(f"§ {paragraph}") or 
                    Chunk.content.contains(f"§{paragraph}")
                ).all()
            
            if not chunks:
                return {
                    "error": f"Параграф {paragraph} не найден в документе {document_id}",
                    "questions": [],
                    "processing_time": time.time() - start_time
                }
            
            # 2. Собираем полный текст параграфа
            full_text = "\n".join([c.content for c in chunks])
            first_chunk = chunks[0]
            
            # 3. Формируем промпт
            prompt = f"""ТЕКСТ ПАРАГРАФА:
{full_text[:4000]}  # Ограничение по токенам

Составь {num_questions} проверочных вопросов по этому тексту.
Вопросы должны быть конкретными, с однозначными ответами.
После вопросов напиши ответы с указанием страниц.

Формат строго соблюдай:"""
            
            # 4. Генерируем вопросы
            raw_response = self.llm.generate(
                prompt=prompt,
                system_message=self.questions_system_prompt,
                temperature=0.3  # Немного креативности для разнообразия вопросов
            )
            
            # 5. Парсим вопросы и ответы
            questions_list = self._parse_questions_response(raw_response, first_chunk)
            
            # 6. СОХРАНЯЕМ КАЖДЫЙ ВОПРОС В БД
            for q in questions_list:
                try:
                    self.save_qa_to_db(
                        query=q["question"],
                        answer=q["answer"],
                        sources=[{"page": q["page"], "chapter": q["chapter"], "paragraph": q["paragraph"]}],
                        mode="question",
                        topic=f"Гл.{chapter}, §{paragraph}"
                    )
                except Exception as e:
                    print(f"⚠️ Ошибка сохранения вопроса: {e}")
                    
            return {
                "questions": questions_list,
                "source_text": full_text[:500] + "...",  # Превью
                "processing_time": time.time() - start_time
            }
            
        finally:
            db.close()
    
    def _parse_fact_answer(self, raw_answer: str) -> Tuple[str, List[Dict]]:
        """Парсит ответ LLM, отделяя текст от ссылок"""
        
        # Пробуем распарсить JSON (если LLM вернула структуру)
        try:
            # Ищем JSON в ответе
            json_match = re.search(r'({.*})', raw_answer, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                answer = data.get('answer', raw_answer)
                sources = data.get('sources', [])
                return answer, sources
        except:
            pass
        
        # Если не JSON, парсим текст
        answer = raw_answer
        
        # Ищем ссылки в формате [Глава X, §Y, стр. Z]
        source_pattern = r'Источник:\s*\[(.*?)\]'
        sources = []
        
        for match in re.finditer(source_pattern, raw_answer):
            source_text = match.group(1)
            sources.append({"reference": source_text})
            # Убираем ссылку из ответа для чистоты
            answer = answer.replace(match.group(0), "")
        
        # Ищем отдельные упоминания страниц
        page_pattern = r'стр\.?\s*(\d+)'
        for match in re.finditer(page_pattern, raw_answer):
            page = match.group(1)
            if not any(s.get('page') == page for s in sources):
                sources.append({"page": page})
        
        return answer.strip(), sources
    
    def _validate_sources(self, sources: List[Dict], actual_metadatas: List[Dict]) -> List[Dict]:
        """
        Проверяет, что указанные источники действительно были в контексте
        """
        valid_sources = []
        
        # Собираем все реальные страницы из метаданных
        valid_pages = set()
        valid_chapters = set()
        valid_paragraphs = set()
        
        for meta in actual_metadatas:
            if meta.get('page_number'):
                valid_pages.add(str(meta['page_number']))
            if meta.get('chapter'):
                valid_chapters.add(meta['chapter'])
            if meta.get('paragraph'):
                valid_paragraphs.add(meta['paragraph'])
        
        for source in sources:
            # Проверяем страницу
            if 'page' in source:
                if str(source['page']) in valid_pages:
                    valid_sources.append(source)
            elif 'reference' in source:
                # Простая проверка: есть ли номер страницы в референсе
                page_match = re.search(r'стр\.?\s*(\d+)', source['reference'])
                if page_match and page_match.group(1) in valid_pages:
                    valid_sources.append(source)
                elif not page_match:  # Если нет страницы, всё равно добавляем
                    valid_sources.append(source)
        
        return valid_sources if valid_sources else sources[:1]  # Хотя бы один
    
    def _parse_questions_response(self, raw_response: str, chunk: Chunk) -> List[Dict]:
        """Парсит ответ с вопросами и ответами"""
        questions = []
        
        # Разделяем на секции ВОПРОСЫ и ОТВЕТЫ
        sections = re.split(r'(ВОПРОСЫ|ОТВЕТЫ)', raw_response, flags=re.IGNORECASE)
        
        questions_text = ""
        answers_text = ""
        
        current_section = None
        for section in sections:
            if section.upper() == "ВОПРОСЫ":
                current_section = "questions"
            elif section.upper() == "ОТВЕТЫ":
                current_section = "answers"
            elif current_section == "questions":
                questions_text = section
            elif current_section == "answers":
                answers_text = section
        
        # Парсим вопросы (нумерованный список)
        q_matches = re.findall(r'\d+\.\s*(.+)', questions_text)
        
        # Парсим ответы с номерами
        a_matches = re.findall(r'\d+\.\s*(.+?)(?:\[стр\.?\s*(\d+)])?\s*(?:\n|$)', answers_text)
        
        # Собираем пары
        for i, q_text in enumerate(q_matches):
            answer_text = ""
            page = chunk.page_number
            
            if i < len(a_matches):
                answer_text = a_matches[i][0].strip()
                if a_matches[i][1]:  # Если указана страница
                    page = int(a_matches[i][1])
            
            questions.append({
                "question": q_text.strip(),
                "answer": answer_text,
                "page": page,
                "chapter": chunk.chapter,
                "paragraph": chunk.paragraph
            })
        
        # Если не удалось распарсить, возвращаем хотя бы 1 вопрос
        if not questions:
            questions.append({
                "question": "Сгенерируйте вопрос по тексту",
                "answer": "Ответ не удалось распарсить",
                "page": chunk.page_number,
                "chapter": chunk.chapter,
                "paragraph": chunk.paragraph
            })
        
        return questions
    

    def save_qa_to_db(self, query: str, answer: str, sources: List[Dict], mode: str, topic: str = ""):
        # Сохраняет вопрос-ответ в БД для будущих тестов """
        from .database import get_db, QALog  # Импортируем новую модель
        
        db = get_db()
        try:
            qa_log = QALog(
                query_text=query,
                answer_text=answer,
                sources_json=json.dumps(sources, ensure_ascii=False),
                mode=mode,
                topic=topic
            )
            db.add(qa_log)
            db.commit()
            return qa_log.id
        finally:
            db.close()