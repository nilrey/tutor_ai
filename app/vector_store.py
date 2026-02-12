import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import uuid
import os

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
                    "section_title": str(chunk.get("section_title", ""))[:200]
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