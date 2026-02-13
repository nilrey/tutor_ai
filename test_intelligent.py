# test_intelligent.py
from app.vector_store import VectorStore
from app.llm_client import LLMClient
from app.intelligent_search import IntelligentSearch

vs = VectorStore()
llm = LLMClient(model_name="gemma3:4b")
searcher = IntelligentSearch(vs, llm)

questions = [
    "Как умер Цезарь?",
    "Кто убил Цезаря?",
    "Когда убили Цезаря?",
    "Почему убили Цезаря?",
    "Что сказал Цезарь перед смертью?"
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"❓ {q}")
    print(f"{'='*60}")
    
    result = searcher.answer_question(q)
    print(f"📖 {result['answer']}")
    if result['sources']:
        print(f"📚 Источник: стр. {result['sources'][0]['page']}")
    print(f"📊 Уверенность: {result['confidence']:.1%}")