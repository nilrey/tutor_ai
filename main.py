from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
from pathlib import Path
import uuid
import os
import warnings
warnings.filterwarnings("ignore")

from app.config import UPLOAD_DIR
from app.database import init_db, Document, Chunk
from app.document_processor import DocumentProcessor
from app.vector_store import VectorStore

# Инициализация
app = FastAPI(title="History AI Tutor - Document Processor")

# Получаем функцию для создания сессий БД
get_db = init_db()  # init_db() возвращает функцию get_db
doc_processor = DocumentProcessor()
vector_store = VectorStore()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Загружает PDF учебник, обрабатывает и индексирует его."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Только PDF файлы поддерживаются")
    
    temp_file_path = None
    
    try:
        # 1. Сохраняем файл
        file_extension = Path(file.filename).suffix
        safe_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = UPLOAD_DIR / safe_filename
        temp_file_path = file_path
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"💾 Файл сохранен: {file_path}")
        
        # 2. Создаем запись в БД - ВЫЗЫВАЕМ get_db() КАК ФУНКЦИЮ
        db = get_db()
        try:
            document = Document(
                filename=file.filename,
                file_path=str(file_path)
            )
            db.add(document)
            db.commit()
            db.refresh(document)
            
            # 3. Обрабатываем документ
            processed_data = doc_processor.process_document(
                file_path=str(file_path),
                filename=file.filename
            )
            
            # 4. Сохраняем чанки в SQL
            for chunk_data in processed_data["chunks"]:
                chunk = Chunk(
                    doc_id=document.id,
                    content=chunk_data["content"],
                    page_number=chunk_data.get("page_number", 1),
                    chapter=chunk_data.get("chapter", ""),
                    paragraph=chunk_data.get("paragraph", ""),
                    section_title=chunk_data.get("section_title", ""),
                    chunk_index=chunk_data["chunk_index"]
                )
                db.add(chunk)
            db.commit()
            
            # 5. Добавляем в векторную БД
            embedding_ids = vector_store.add_chunks(
                processed_data["chunks"],
                document.id
            )
            
            # 6. Обновляем чанки с ID эмбеддингов
            if embedding_ids:
                for chunk_data, emb_id in zip(processed_data["chunks"], embedding_ids):
                    db.query(Chunk).filter(
                        Chunk.doc_id == document.id,
                        Chunk.chunk_index == chunk_data["chunk_index"]
                    ).update({"embedding_id": emb_id})
                db.commit()
            
            # 7. Обновляем статистику документа
            document.total_chunks = len(processed_data["chunks"])
            db.commit()
            
            return JSONResponse({
                "status": "success",
                "document_id": document.id,
                "filename": file.filename,
                "total_pages": processed_data["total_pages"],
                "total_chunks": len(processed_data["chunks"]),
                "chapters_found": len(processed_data.get("chapters", [])),
                "paragraphs_found": len(processed_data.get("paragraphs", [])),
                "message": "Учебник успешно загружен и проиндексирован"
            })
            
        finally:
            db.close()
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Очищаем файл при ошибке
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink()
        raise HTTPException(500, f"Ошибка при обработке: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Возвращает статистику по загруженным документам"""
    db = get_db()  # БЫЛО: db = db_session()
    
    try:
        docs = db.query(Document).all()
        total_chunks = db.query(Chunk).count()
        
        vector_stats = vector_store.get_collection_stats()
        
        return {
            "documents": [
                {
                    "id": d.id,
                    "filename": d.filename,
                    "upload_date": d.upload_date.isoformat() if d.upload_date else None,
                    "chunks": d.total_chunks
                }
                for d in docs
            ],
            "total_documents": len(docs),
            "total_chunks_sql": total_chunks,
            "vector_db": vector_stats
        }
    finally:
        db.close()  # Важно закрывать сессию!

@app.get("/documents/{doc_id}/chunks")
async def get_document_chunks(doc_id: int, skip: int = 0, limit: int = 10):
    """Получает чанки документа для просмотра"""
    db = get_db()  # БЫЛО: db = db_session()
    
    try:
        chunks = db.query(Chunk).filter(
            Chunk.doc_id == doc_id
        ).offset(skip).limit(limit).all()
        
        total = db.query(Chunk).filter(Chunk.doc_id == doc_id).count()
        
        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "chunks": [
                {
                    "id": c.id,
                    "content_preview": c.content[:200] + "..." if len(c.content) > 200 else c.content,
                    "page": c.page_number,
                    "chapter": c.chapter,
                    "paragraph": c.paragraph,
                    "title": c.section_title
                }
                for c in chunks
            ]
        }
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    """Действия при запуске"""
    print("🚀 Запуск History AI Tutor")
    print(f"📁 Директория загрузок: {UPLOAD_DIR}")
    print(f"🗄️ Векторная БД: {vector_store.get_collection_stats()}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)