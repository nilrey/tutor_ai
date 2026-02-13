# test_sql.py
from app.database import get_db, Chunk
from sqlalchemy import text

db = get_db()
try:
    # Прямой SQL запрос
    result = db.execute(
        text("""
        SELECT page_number, chapter, paragraph, substr(content, 1, 300) as preview 
        FROM chunks 
        WHERE content LIKE '%Цез%' 
           OR content LIKE '%Юлий%' 
           OR content LIKE '%кесар%'
           OR content LIKE '%Caesar%'
        ORDER BY page_number
        """)
    ).fetchall()
    
    print(f"🔍 Найдено: {len(result)} чанков\n")
    for row in result:
        print(f"📄 Стр. {row[0]} | {row[1]} {row[2]}")
        print(f"{row[3]}...\n")
        
finally:
    db.close()