# test_search.py
from app.vector_store import VectorStore

print("🔍 Инициализация VectorStore...")
vs = VectorStore()

# Вместо _load_embedding_model() используем прямой доступ
if vs.embedding_model is None:
    print("🔄 Загружаем модель эмбеддингов...")
    # Принудительно вызываем метод, который загружает модель
    vs.add_chunks([], 0)  # Пустой вызов для инициализации

# Тестовый запрос
queries = ["реформа римского календаря"]

for q in queries:
    print(f"\n🔍 Поиск: {q}")
    results = vs.search(q, n_results=3)
    
    if results and results.get('documents') and results['documents'][0]:
        print(f"✅ Найдено: {len(results['documents'][0])} чанков")
        for i, doc in enumerate(results['documents'][0][:2]):
            print(f"\n--- Чанк {i+1} ---")
            print(f"Страница: {results['metadatas'][0][i].get('page_number', '?')}")
            print(f"Текст: {doc}...")
    else:
        print("❌ Ничего не найдено")