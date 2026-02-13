from app.fact_retrieval import FactRetrievalEngine
from app.vector_store import VectorStore

vs = VectorStore()
engine = FactRetrievalEngine(vs)

queries = [
    "Как умер Цезарь?",
    "Кто убил Цезаря?",
    "В каком году началась Вторая мировая война?",
    "Кто был Наполеон?"
]

for q in queries:
    print("\n============================")
    print("QUERY:", q)
    results = engine.retrieve(q)

    for i, r in enumerate(results):
        print(f"\n--- RESULT {i+1} ---")
        print("SCORE:", r["score"])
        print("SOURCE:", r["source"])
        print("META:", r["metadata"])
        print("TEXT:", r["content"][:300])
