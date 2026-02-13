# intelligent_search.py
from typing import List, Dict, Any, Optional
import time

class IntelligentSearch:
    """
    Интеллектуальный поиск с пониманием контекста через LLM
    """
    
    def __init__(self, vector_store, llm_client):
        self.vs = vector_store
        self.llm = llm_client
    
    def expand_query_with_llm(self, query: str) -> List[str]:
        """
        Использует LLM для интеллектуального расширения запроса
        """
        prompt = f"""Переформулируй вопрос в 3 разных варианта для поиска в учебнике истории.
Сохрани смысл, но используй разные формулировки.

Исходный вопрос: {query}

Пример:
Вопрос: "Как умер Цезарь?"
Варианты:
- смерть Гая Юлия Цезаря
- убийство Цезаря
- обстоятельства гибели Цезаря

Теперь для твоего вопроса:"""

        try:
            response = self.llm.generate(
                prompt=prompt,
                system_message="Ты помогаешь улучшить поиск. Отвечай кратко, только варианты.",
                temperature=0.3
            )
            
            # Парсим ответ
            variants = [query]  # Оригинал всегда включаем
            for line in response.split('\n'):
                line = line.strip()
                # Убираем маркеры списка и пустые строки
                if line and not line.startswith(('Вариант', '-', '•', '*')):
                    # Убираем нумерацию если есть
                    if line[0].isdigit() and line[1:].startswith('. '):
                        line = line[3:]
                    variants.append(line)
            
            return variants[:4]  # Не больше 4 вариантов
            
        except Exception as e:
            print(f"⚠️ Ошибка расширения запроса: {e}")
            return [query]
    
    def intelligent_search(self, query: str, n_results: int = 3) -> List[Dict]:
        """
        Интеллектуальный поиск с переформулировкой запроса
        """
        # 1. Получаем разные формулировки того же вопроса
        variants = self.expand_query_with_llm(query)
        print(f"🔄 Варианты запроса: {variants}")
        
        # 2. Ищем по каждому варианту
        all_results = []
        seen_chunks = set()
        
        for variant in variants:
            # Векторный поиск
            results = self.vs.search(variant, n_results=n_results * 2)
            
            if results and results.get('documents'):
                for i, doc in enumerate(results['documents'][0]):
                    # Создаем уникальный ID для чанка
                    meta = results['metadatas'][0][i]
                    chunk_id = f"{meta.get('doc_id', '')}_{meta.get('page_number', '')}_{i}"
                    
                    if chunk_id not in seen_chunks:
                        seen_chunks.add(chunk_id)
                        all_results.append({
                            'content': doc,
                            'metadata': meta,
                            'distance': results['distances'][0][i] if results.get('distances') else 1.0,
                            'query': variant
                        })
        
        # 3. Сортируем по близости (меньше расстояние = лучше)
        all_results.sort(key=lambda x: x['distance'])
        
        return all_results[:n_results]
    
    def extract_answer(self, query: str, chunks: List[Dict]) -> str:
        """
        Извлекает ответ из найденных чанков с пониманием контекста
        """
        if not chunks:
            return "Информация не найдена"
        
        # Собираем контекст
        context_parts = []
        for i, chunk in enumerate(chunks[:3]):  # Максимум 3 чанка
            page = chunk['metadata'].get('page_number', '?')
            text = chunk['content'][:1000]  # Ограничиваем длину
            context_parts.append(f"[Страница {page}]\n{text}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""Прочитай фрагменты учебника и ответь на вопрос.

Вопрос: {query}

Фрагменты учебника:
{context}

Ответь на вопрос, используя ТОЛЬКО информацию из текста.
Если точного ответа нет, но есть связанная информация - напиши что нашел.
Если информации нет совсем - скажи "Информация отсутствует в учебнике".

Ответ:"""
        
        try:
            answer = self.llm.generate(
                prompt=prompt,
                system_message="Ты отвечаешь строго по тексту учебника.",
                temperature=0.0
            )
            return answer.strip()
        except Exception as e:
            print(f"⚠️ Ошибка извлечения ответа: {e}")
            # Возвращаем первый чанк как запасной вариант
            return chunks[0]['content'][:300] + "..."
    
    def answer_question(self, query: str) -> Dict[str, Any]:
        """
        Полный цикл ответа на вопрос
        """
        start_time = time.time()
        
        # 1. Интеллектуальный поиск
        chunks = self.intelligent_search(query, n_results=3)
        
        # 2. Извлечение ответа
        answer = self.extract_answer(query, chunks)
        
        # 3. Подготовка источников
        sources = []
        for chunk in chunks[:2]:  # Топ-2 источника
            sources.append({
                'page': chunk['metadata'].get('page_number'),
                'chapter': chunk['metadata'].get('chapter'),
                'paragraph': chunk['metadata'].get('paragraph'),
                'text_preview': chunk['content'][:150] + '...'
            })
        
        processing_time = time.time() - start_time
        
        return {
            'answer': answer,
            'sources': sources,
            'confidence': 1.0 - chunks[0]['distance'] if chunks else 0,
            'processing_time': processing_time
        }