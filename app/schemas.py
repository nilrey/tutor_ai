from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QuestionRequest(BaseModel):
    """Запрос на фактологический вопрос"""
    query: str
    document_id: Optional[int] = None  # Если None - ищем по всем
    top_k: int = 2

class QuestionResponse(BaseModel):
    """Ответ на фактологический вопрос"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: Optional[float] = None
    processing_time: Optional[float] = None

class GenerateQuestionsRequest(BaseModel):
    """Запрос на генерацию вопросов по параграфу"""
    document_id: int
    chapter: str
    paragraph: str
    num_questions: int = 5

class QuestionItem(BaseModel):
    """Один вопрос с ответом и источником"""
    question: str
    answer: str
    page: int
    chapter: str
    paragraph: str

class GenerateQuestionsResponse(BaseModel):
    """Ответ с вопросами"""
    questions: List[QuestionItem]
    source_text: str  # Для отладки