# test_ollama_connection.py
import requests

# Проверяем доступность модели через API
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "gemma3:4b",
        "prompt": "Скажи 'Привет' одним словом",
        "stream": False
    }
)
print(response.json()['response'])