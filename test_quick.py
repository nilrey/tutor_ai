import requests
import time

print("🔍 Тестируем подключение к Ollama...")

# 1. Проверяем сервер
try:
    r = requests.get("http://localhost:11434/api/tags", timeout=5)
    print(f"✅ Ollama сервер доступен")
    models = r.json().get('models', [])
    for m in models:
        print(f"   - {m['name']}")
except Exception as e:
    print(f"❌ Ollama не отвечает: {e}")
    print("   Запустите Ollama из меню Пуск или командой 'ollama serve'")
    exit()

# 2. Тестируем модель
model_name = "gemma3:4b"  # или llama3:latest
print(f"\n🔄 Тестируем модель {model_name}...")

try:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model_name,
            "prompt": "Ответь одним словом: 2+2=?",
            "stream": False,
            "options": {
                "num_predict": 10,
                "temperature": 0
            }
        },
        timeout=30
    )
    
    if response.status_code == 200:
        answer = response.json()['response']
        print(f"✅ Модель отвечает: {answer}")
    else:
        print(f"❌ Ошибка: {response.status_code}")
        
except Exception as e:
    print(f"❌ Ошибка при запросе: {e}")