import os
from pathlib import Path

# Базовая директория проекта
BASE_DIR = Path(__file__).parent.parent

# Директория для загрузок
UPLOAD_DIR = BASE_DIR / "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Параметры чанкинга
CHUNK_SIZE = 1000  # символов на чанк
CHUNK_OVERLAP = 200  # перекрытие между чанками

# Настройки БД
DATABASE_URL = f"sqlite:///{BASE_DIR}/history_tutor.db"

# Настройки векторной БД (для FAISS)
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"  # оставляем для совместимости
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# Модель для эмбеддингов
# EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # Поддерживает русский
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"