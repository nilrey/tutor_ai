from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json

from .config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    total_chunks = Column(Integer, default=0)
    
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"
    
    id = Column(Integer, primary_key=True)
    doc_id = Column(Integer, ForeignKey("documents.id"))
    content = Column(Text, nullable=False)
    page_number = Column(Integer)
    chapter = Column(String(200))
    paragraph = Column(String(200))
    section_title = Column(String(300))
    chunk_index = Column(Integer)
    embedding_id = Column(String(100))  # ID в векторной БД
    
    document = relationship("Document", back_populates="chunks")

def get_db():
    """Создает и возвращает новую сессию БД"""
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.close()
        raise

def init_db():
    """Инициализация БД - создает таблицы и возвращает функцию для получения сессий"""
    Base.metadata.create_all(bind=engine)
    return get_db  # Возвращаем функцию, а не класс