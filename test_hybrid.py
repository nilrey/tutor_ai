# test_hybrid.py
from app.vector_store import VectorStore

vs = VectorStore()
vs._load_embedding_model()

queries = [
    "Когда умер Цезарь?"
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"🔍 Запрос: {query}")
    print(f"{'='*60}")
    
    results = vs.hybrid_search(query, n_results=3)
    
    for i, r in enumerate(results):
        print(f"\n--- Результат {i+1} (источник: {r['source']}, скор: {r['final_score']:.2f}) ---")
        print(f"📄 Стр. {r['metadata'].get('page_number', '?')}")
        if r.get('keywords'):
            print(f"🔑 Найдены: {r['keywords']}")
        print(f"Текст: {r['content']}")