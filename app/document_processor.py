import re
from pathlib import Path
import PyPDF2
import pdfplumber
from typing import List, Dict, Any, Tuple
import tiktoken

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Для подсчета токенов (опционально)
        self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict[int, str]]:
        """
        Извлекает текст из PDF.
        Возвращает полный текст и словарь {страница: текст_страницы}
        """
        full_text = []
        pages_text = {}
        
        # Пробуем pdfplumber (лучше для сложной верстки)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    pages_text[i] = text
                    full_text.append(text)
            print(f"✅ PDFplumber: извлечено {len(pages_text)} страниц")
        except Exception as e:
            print(f"⚠️ PDFplumber ошибка: {e}, пробуем PyPDF2")
            
            # Fallback на PyPDF2
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for i, page in enumerate(reader.pages, 1):
                    text = page.extract_text() or ""
                    pages_text[i] = text
                    full_text.append(text)
        
        return "\n".join(full_text), pages_text
    
    def detect_structure(self, text: str, pages_text: Dict[int, str]) -> Dict[str, Any]:
        """
        Пытается определить структуру учебника:
        главы, параграфы, заголовки.
        """
        structure = {
            "chapters": [],
            "paragraphs": []
        }
        
        # Паттерны для русского языка
        chapter_patterns = [
            r'Глава\s*(\d+|[IVXLCDM]+)\.?\s*(.*?)(?=\n|$)',
            r'Раздел\s*(\d+|[IVXLCDM]+)\.?\s*(.*?)(?=\n|$)',
            r'Часть\s*(\d+|[IVXLCDM]+)\.?\s*(.*?)(?=\n|$)'
        ]
        
        paragraph_patterns = [
            r'§\s*(\d+|[IVXLCDM]+)\.?\s*(.*?)(?=\n|$)',
            r'Параграф\s*(\d+|[IVXLCDM]+)\.?\s*(.*?)(?=\n|$)'
        ]
        
        # Ищем главы
        for pattern in chapter_patterns:
            chapters = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if chapters:
                structure["chapters"] = [{"number": num, "title": title.strip()} for num, title in chapters]
                break
        
        # Ищем параграфы
        for pattern in paragraph_patterns:
            paragraphs = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if paragraphs:
                structure["paragraphs"] = [{"number": num, "title": title.strip()} for num, title in paragraphs]
                break
        
        return structure
    
    def create_chunks(self, text: str, pages_text: Dict[int, str], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Разбивает текст на чанки с умным делением по параграфам.
        """
        chunks = []
        
        # Сначала пробуем разделить по параграфам (§)
        paragraphs = re.split(r'(?=§\s*\d+)|(?=\n\s*\n)', text)
        
        current_chunk = ""
        current_chunk_start_page = 1
        chunk_index = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Определяем страницу для этого параграфа
            page_num = self._find_page_for_text(paragraph, pages_text)
            
            # Если текущий чанк + новый параграф не превышают размер
            if len(current_chunk) + len(paragraph) < self.chunk_size:
                current_chunk += paragraph + "\n"
            else:
                # Сохраняем текущий чанк
                if current_chunk:
                    chunk_metadata = self._extract_chunk_metadata(current_chunk, metadata)
                    chunks.append({
                        "content": current_chunk.strip(),
                        "page_number": current_chunk_start_page,
                        "chapter": chunk_metadata.get("chapter", ""),
                        "paragraph": chunk_metadata.get("paragraph", ""),
                        "section_title": chunk_metadata.get("title", ""),
                        "chunk_index": chunk_index
                    })
                    chunk_index += 1
                
                # Начинаем новый чанк
                current_chunk = paragraph + "\n"
                current_chunk_start_page = page_num or current_chunk_start_page
        
        # Добавляем последний чанк
        if current_chunk:
            chunk_metadata = self._extract_chunk_metadata(current_chunk, metadata)
            chunks.append({
                "content": current_chunk.strip(),
                "page_number": current_chunk_start_page,
                "chapter": chunk_metadata.get("chapter", ""),
                "paragraph": chunk_metadata.get("paragraph", ""),
                "section_title": chunk_metadata.get("title", ""),
                "chunk_index": chunk_index
            })
        
        print(f"📄 Создано {len(chunks)} чанков")
        return chunks
    
    def _find_page_for_text(self, text: str, pages_text: Dict[int, str]) -> int:
        """Находит страницу, на которой встречается текст"""
        # Берем первые 50 символов текста для поиска
        sample = text[:50].strip()
        for page_num, page_content in pages_text.items():
            if sample in page_content:
                return page_num
        return 1  # По умолчанию страница 1
    
    def _extract_chunk_metadata(self, chunk: str, global_metadata: Dict[str, Any]) -> Dict[str, str]:
        """Извлекает метаданные из текста чанка"""
        metadata = {
            "chapter": "",
            "paragraph": "",
            "title": ""
        }
        
        # Ищем главу в тексте чанка
        chapter_match = re.search(r'Глава\s*(\d+|[IVXLCDM]+)', chunk, re.IGNORECASE)
        if chapter_match:
            metadata["chapter"] = chapter_match.group(0)
        
        # Ищем параграф
        paragraph_match = re.search(r'§\s*(\d+|[IVXLCDM]+)', chunk, re.IGNORECASE)
        if paragraph_match:
            metadata["paragraph"] = paragraph_match.group(0)
            
            # Ищем заголовок после параграфа
            title_match = re.search(r'§\s*\d+\.?\s*(.*?)(?=\n|$)', chunk)
            if title_match:
                metadata["title"] = title_match.group(1).strip()
        
        return metadata
    
    def process_document(self, file_path: str, filename: str) -> Dict[str, Any]:
        """
        Полный цикл обработки документа.
        """
        print(f"🔄 Начинаем обработку: {filename}")
        
        # 1. Извлекаем текст
        full_text, pages_text = self.extract_text_from_pdf(file_path)
        print(f"📖 Всего символов: {len(full_text)}")
        
        # 2. Определяем структуру
        structure = self.detect_structure(full_text, pages_text)
        print(f"📚 Найдено глав: {len(structure['chapters'])}")
        print(f"📑 Найдено параграфов: {len(structure['paragraphs'])}")
        
        # 3. Создаем чанки
        chunks = self.create_chunks(full_text, pages_text, structure)
        
        return {
            "filename": filename,
            "total_chars": len(full_text),
            "total_pages": len(pages_text),
            "chapters": structure["chapters"],
            "paragraphs": structure["paragraphs"],
            "chunks": chunks
        }