# test_exact_search.py
from app.database import get_db, Chunk
from sqlalchemy import or_

db = get_db()
try:
    # Ищем точные упоминания Цезаря
    chunks = db.query(Chunk).filter(
        or_(
            Chunk.content.ilike('%Цезарь%'),
            Chunk.content.ilike('%Гай Юлий%'),
            Chunk.content.ilike('%Юлий Цезарь%'),
            Chunk.content.ilike('%кесарь%')
        )
    ).all()
    
    print(f"🔍 Найдено чанков с Цезарем: {len(chunks)}")
    
    for chunk in chunks[:5]:
        print(f"\n--- Страница {chunk.page_number} ---")
        print(chunk.content[:500])
        print("...")
        
finally:
    db.close()