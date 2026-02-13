import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import uuid
import os
import re
from collections import Counter

from .config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL

class VectorStore:
    def __init__(self):
        """Инициализация ChromaDB и модели эмбеддингов"""
        # Создаем директорию для ChromaDB если её нет
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        
        # Инициализируем ChromaDB
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(CHROMA_PERSIST_DIR)
            )
            print("✅ ChromaDB клиент инициализирован")
        except Exception as e:
            print(f"❌ Ошибка инициализации ChromaDB: {e}")
            raise
        
        # Пробуем получить существующую коллекцию или создаем новую
        try:
            self.collection = self.chroma_client.get_collection("history_textbooks")
            print(f"📚 Загружена существующая коллекция, чанков: {self.collection.count()}")
        except:
            self.collection = self.chroma_client.create_collection(
                name="history_textbooks",
                metadata={"hnsw:space": "cosine"}
            )
            print("✅ Создана новая коллекция")
        
        # Инициализируем модель для эмбеддингов
        print(f"🔄 Загружаем модель эмбеддингов: {EMBEDDING_MODEL}")
        try:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            print(f"✅ Модель загружена, размерность: {self.embedding_model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки модели: {e}")
            print("🔄 Пробуем загрузить английскую модель...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Загружена английская модель")
    
    def add_chunks(self, chunks: List[Dict[str, Any]], doc_id: int) -> List[str]:
        """
        Добавляет чанки в векторную БД.
        Возвращает список ID эмбеддингов.
        """
        if not chunks:
            return []
        
        embeddings = []
        metadatas = []
        ids = []
        documents = []
        
        print(f"🔄 Добавляем {len(chunks)} чанков в ChromaDB...")
        
        for i, chunk in enumerate(chunks):
            try:
                # Генерируем уникальный ID
                chunk_id = f"doc{doc_id}_chunk{i}_{uuid.uuid4().hex[:8]}"
                
                # Создаем эмбеддинг
                embedding = self.embedding_model.encode(chunk["content"]).tolist()
                
                # Подготавливаем метаданные (все значения должны быть строками)
                metadata = {
                    "doc_id": str(doc_id),
                    "chunk_index": str(i),
                    "page_number": str(chunk.get("page_number", 1)),
                    "chapter": str(chunk.get("chapter", ""))[:100],
                    "paragraph": str(chunk.get("paragraph", ""))[:100],
                    "section_title": str(chunk.get("section_title", ""))[:200],
                    "id": str(i)  # Добавляем ID для поиска
                }
                
                embeddings.append(embedding)
                metadatas.append(metadata)
                ids.append(chunk_id)
                documents.append(chunk["content"][:1000])  # Ограничиваем длину для ChromaDB
                
                if i % 50 == 0 and i > 0:
                    print(f"  ⏳ Обработано {i}/{len(chunks)} чанков")
                    
            except Exception as e:
                print(f"⚠️ Ошибка подготовки чанка {i}: {e}")
        
        # Добавляем в коллекцию батчами по 100
        batch_size = 100
        added_count = 0
        
        for i in range(0, len(embeddings), batch_size):
            try:
                batch_end = min(i + batch_size, len(embeddings))
                self.collection.add(
                    embeddings=embeddings[i:batch_end],
                    metadatas=metadatas[i:batch_end],
                    ids=ids[i:batch_end],
                    documents=documents[i:batch_end]
                )
                added_count += (batch_end - i)
                print(f"  ✓ Добавлен батч {i//batch_size + 1}/{(len(embeddings)-1)//batch_size + 1}")
            except Exception as e:
                print(f"  ✗ Ошибка добавления батча: {e}")
                # Пробуем добавить по одному
                for j in range(i, min(i + batch_size, len(embeddings))):
                    try:
                        self.collection.add(
                            embeddings=[embeddings[j]],
                            metadatas=[metadatas[j]],
                            ids=[ids[j]],
                            documents=[documents[j]]
                        )
                        added_count += 1
                    except Exception as e2:
                        print(f"    ✗ Ошибка добавления чанка {j}: {e2}")
        
        print(f"✅ Успешно добавлено {added_count}/{len(chunks)} чанков в ChromaDB")
        return ids[:added_count]
    
    def get_collection_stats(self):
        """Возвращает статистику коллекции"""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection.name,
                "status": "active"
            }
        except Exception as e:
            return {
                "total_chunks": 0,
                "collection_name": "history_textbooks",
                "status": f"error: {e}"
            }
    
    def search(self, query: str, n_results: int = 5) -> Optional[Dict]:
        """Поиск похожих чанков"""
        try:
            # Создаем эмбеддинг запроса
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Ищем похожие чанки
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["metadatas", "documents", "distances"]
            )
            
            return results
        except Exception as e:
            print(f"❌ Ошибка поиска: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def delete_document(self, doc_id: int):
        """Удаляет все чанки документа"""
        try:
            self.collection.delete(
                where={"doc_id": str(doc_id)}
            )
            print(f"✅ Удалены чанки документа {doc_id} из ChromaDB")
        except Exception as e:
            print(f"❌ Ошибка удаления документа {doc_id}: {e}")

    def _load_embedding_model(self):
        """Ленивая загрузка модели эмбеддингов"""
        if self.embedding_model is None:
            print(f"🔄 Загружаем модель эмбеддингов: {EMBEDDING_MODEL}")
            try:
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
                print(f"✅ Модель загружена, размерность: {self.embedding_model.get_sentence_embedding_dimension()}")
            except Exception as e:
                print(f"⚠️ Ошибка загрузки модели: {e}")
                print("🔄 Пробуем загрузить английскую модель...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Загружена английская модель")
        return self.embedding_model
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Извлекает ключевые слова из запроса
        """
        # Приводим к нижнему регистру
        query_lower = query.lower()
        
        # Удаляем знаки препинания
        query_lower = re.sub(r'[^\w\s]', ' ', query_lower)
        
        # Разбиваем на слова
        words = query_lower.split()
        
        # Стоп-слова (короткие и частотные)
        stop_words = {'когда', 'где', 'какой', 'какая', 'какое', 'какие', 'что', 'кто', 
                     'как', 'почему', 'зачем', 'сколько', 'этот', 'эта', 'это', 'эти',
                     'весь', 'вся', 'все', 'был', 'была', 'было', 'были', 'при', 'для',
                     'чтобы', 'чрез', 'через', 'около', 'почти', 'уже', 'еще', 'ещё'}
        
        # Оставляем слова длиннее 3 символов и не в стоп-листе
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        # Добавляем вариации для имен (Цезарь -> цезар, юлий)
        variations = []
        for word in keywords:
            if word in ['цезарь', 'цезаря', 'цезарю', 'цезарем']:
                variations.extend(['цезар', 'юлий'])
            if word in ['юлий', 'юлия']:
                variations.append('юлий')
                
        keywords.extend(variations)
        
        return list(set(keywords))  # Убираем дубликаты
    
    def _keyword_search_sql(self, keywords: List[str], n_results: int) -> List[Dict]:
        """
        Поиск по ключевым словам через SQL с ранжированием
        """
        from .database import get_db, Chunk
        from sqlalchemy import or_, and_
        
        db = get_db()
        try:
            if not keywords:
                return []
            
            # Создаем условия для каждого ключевого слова
            conditions = []
            for word in keywords:
                # Ищем разные формы слова
                conditions.append(Chunk.content.ilike(f'%{word}%'))
                conditions.append(Chunk.content.ilike(f'%{word.capitalize()}%'))
            
            # Выполняем поиск
            chunks = db.query(Chunk).filter(
                or_(*conditions)
            ).limit(n_results * 2).all()  # Берем с запасом
            
            # Ранжируем по частоте вхождений
            ranked_chunks = []
            for chunk in chunks:
                content_lower = chunk.content.lower()
                score = 0
                
                # Считаем сколько ключевых слов найдено
                found_keywords = []
                for word in keywords:
                    if word in content_lower:
                        score += 1
                        found_keywords.append(word)
                        # Дополнительный вес за точное совпадение
                        if f" {word} " in f" {content_lower} ":
                            score += 1
                
                # Особый вес для имен
                if 'юлий' in found_keywords or 'цезар' in found_keywords:
                    score += 3
                
                if score > 0:
                    ranked_chunks.append({
                        'content': chunk.content,
                        'metadata': {
                            'doc_id': str(chunk.doc_id),
                            'page_number': str(chunk.page_number),
                            'chapter': chunk.chapter or '',
                            'paragraph': chunk.paragraph or '',
                            'id': chunk.id
                        },
                        'score': score,
                        'source': 'keyword',
                        'keywords_found': found_keywords
                    })
            
            # Сортируем по убыванию скора
            ranked_chunks.sort(key=lambda x: x['score'], reverse=True)
            
            return ranked_chunks[:n_results]
            
        finally:
            db.close()
    
    def hybrid_search(self, query: str, n_results: int = 5, vector_weight: float = 0.4):
        """
        Улучшенный гибридный поиск
        """
        # 1. Извлекаем ключевые слова
        keywords = self._extract_keywords(query)
        print(f"🔑 Ключевые слова: {keywords}")
        
        # 2. Поиск по ключевым словам (SQL)
        keyword_results = self._keyword_search_sql(keywords, n_results)
        
        # 3. Векторный поиск
        vector_results = self.search(query, n_results=n_results * 2)
        
        # 4. Комбинируем результаты
        combined_chunks = []
        seen_ids = set()
        
        # Сначала добавляем результаты из ключевого поиска (высокий приоритет для имен)
        for chunk in keyword_results:
            chunk_id = chunk['metadata'].get('id')
            if chunk_id not in seen_ids:
                # Нормализуем score в диапазон 0-1
                max_keyword_score = max([c['score'] for c in keyword_results]) if keyword_results else 1
                norm_score = chunk['score'] / max_keyword_score
                
                combined_chunks.append({
                    'content': chunk['content'],
                    'metadata': chunk['metadata'],
                    'score': norm_score,
                    'source': 'keyword',
                    'keywords': chunk.get('keywords_found', [])
                })
                seen_ids.add(chunk_id)
        
        # Затем добавляем результаты из векторного поиска
        if vector_results and vector_results.get('documents'):
            for i, doc in enumerate(vector_results['documents'][0]):
                meta = vector_results['metadatas'][0][i]
                chunk_id = meta.get('id', i)
                
                if chunk_id not in seen_ids:
                    distance = vector_results['distances'][0][i] if vector_results.get('distances') else 0
                    vector_score = 1.0 - min(distance, 1.0)  # Нормализуем
                    
                    combined_chunks.append({
                        'content': doc,
                        'metadata': meta,
                        'score': vector_score,
                        'source': 'vector'
                    })
                    seen_ids.add(chunk_id)
        
        # 5. Финальная сортировка с весами
        for chunk in combined_chunks:
            if chunk['source'] == 'keyword':
                # Для ключевых слов оставляем высокий вес
                chunk['final_score'] = chunk['score']
            else:
                # Для векторных - с коэффициентом
                chunk['final_score'] = chunk['score'] * vector_weight
        
        combined_chunks.sort(key=lambda x: x['final_score'], reverse=True)
        
        # 6. Возвращаем топ результатов
        return combined_chunks[:n_results]