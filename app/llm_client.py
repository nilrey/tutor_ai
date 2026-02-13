import requests
import json
from typing import Optional, Dict, Any
import time

class LLMClient:
    """Клиент для работы с локальными моделями через Ollama"""
    
    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434"):
        """
        Инициализация клиента Ollama
        
        Args:
            model_name: Имя модели в Ollama (llama3, mistral, gemma, etc.)
            base_url: Адрес Ollama API
        """
        self.model_name = model_name
        self.base_url = base_url
        self.use_mock = False
        
        # Проверяем доступность Ollama
        try:
            response = requests.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m['name'] for m in models]
                print(f"✅ Ollama доступна. Модели: {available_models}")
                
                # Проверяем, есть ли запрошенная модель
                if not any(model_name in m for m in available_models):
                    print(f"⚠️ Модель {model_name} не найдена. Доступны: {available_models}")
                    if available_models:
                        self.model_name = available_models[0]
                        print(f"🔄 Используем {self.model_name} вместо {model_name}")
            else:
                print("⚠️ Ollama не отвечает, используется заглушка")
                self.use_mock = True
        except Exception as e:
            print(f"⚠️ Ollama недоступна: {e}")
            print("🔄 Используется режим заглушки (mock)")
            self.use_mock = True
    
    def generate(self, prompt: str, system_message: str = "", temperature: float = 0.0) -> str:
        """
        Отправляет запрос в локальную модель Ollama
        """
        if self.use_mock:
            return self._mock_response(prompt)
        
        try:
            # Формируем запрос для Ollama
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "num_predict": 300,  # БЫЛО 1000 - слишком много!
                    "num_ctx": 2048,     # Контекст поменьше
                }
            }
            
            # Отправляем запрос
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60  # Таймаут 60 секунд
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['message']['content']
            else:
                print(f"❌ Ollama ошибка: {response.status_code} - {response.text}")
                return f"Ошибка модели: {response.status_code}"
                
        except requests.exceptions.Timeout:
            print("❌ Таймаут Ollama (модель слишком долго думает)")
            return "Извините, модель слишком долго обрабатывает запрос. Попробуйте упростить вопрос."
        except Exception as e:
            print(f"❌ Ошибка Ollama: {e}")
            return f"Ошибка при обращении к Ollama: {str(e)}"
    
    def _mock_response(self, prompt: str) -> str:
        """Заглушка для тестирования без Ollama"""
        print("⚠️ Используется режим заглушки (mock)")
        
        if "факт" in prompt.lower() or "вопрос" in prompt.lower():
            return """На основе предоставленного контекста:

1 сентября 1939 года.

Источник: [Глава 5, §2, стр. 112]"""
        else:
            return """ВОПРОСЫ:
1. В каком году началась Вторая мировая война?
2. Какое событие считается началом войны?
3. Кто был главой СССР в 1939 году?

ОТВЕТЫ:
1. 1939 год [стр. 112]
2. Нападение Германии на Польшу [стр. 112]
3. Иосиф Сталин [стр. 115]"""
    
    def is_available(self) -> bool:
        """Проверяет доступность Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> list:
        """Возвращает список доступных моделей"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                return [m['name'] for m in response.json().get('models', [])]
        except:
            pass
        return []