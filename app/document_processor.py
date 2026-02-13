# app/document_processor.py
from pathlib import Path
from typing import List, Dict
import re
import pdfplumber

class DocumentProcessor:
    """
    Новый процессор документов:
    - чистка текста
    - удаление шума
    - semantic chunking
    - абзацная логика
    - подготовка данных под fact-based RAG
    """

    def __init__(self, min_chunk_len: int = 300, max_chunk_len: int = 1200):
        self.min_chunk_len = min_chunk_len
        self.max_chunk_len = max_chunk_len

    # ---------------- CLEANING ----------------

    def clean_text(self, text: str) -> str:
        lines = text.split("\n")
        clean_lines = []

        noise_prefixes = (
            "§", "ГЛАВА", "Глава", "Рис", "Рисунок", "Таблица",
            "Вопросы", "Задания", "?", "•", "—"
        )

        for line in lines:
            line = line.strip()

            # слишком короткие строки
            if len(line) < 25:
                continue

            # служебные элементы
            if line.startswith(noise_prefixes):
                continue

            # только цифры
            if re.fullmatch(r"[0-9\s]+", line):
                continue

            # мусорные символы
            if len(re.findall(r"[А-Яа-я]", line)) < 10:
                continue

            clean_lines.append(line)

        return "\n".join(clean_lines)

    # ---------------- NORMALIZATION ----------------

    def normalize_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"-\s+", "", text)  # переносы слов
        return text.strip()

    # ---------------- SEMANTIC CHUNKING ----------------

    def semantic_chunking(self, text: str) -> List[str]:
        paragraphs = re.split(r"\n{2,}", text)

        chunks = []
        buffer = ""

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue

            if len(buffer) + len(p) < self.max_chunk_len:
                buffer += " " + p
            else:
                if len(buffer) >= self.min_chunk_len:
                    chunks.append(buffer.strip())
                buffer = p

        if buffer and len(buffer) >= self.min_chunk_len:
            chunks.append(buffer.strip())

        return chunks

    # ---------------- MAIN PIPELINE ----------------

    def process_document(self, file_path: str, filename: str) -> Dict:
        pages_text = []

        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                raw_text = page.extract_text() or ""

                cleaned = self.clean_text(raw_text)
                normalized = self.normalize_text(cleaned)

                if normalized:
                    pages_text.append({
                        "page": i + 1,
                        "text": normalized
                    })

        all_text = "\n\n".join([p["text"] for p in pages_text])

        chunks = self.semantic_chunking(all_text)

        processed_chunks = []
        for idx, ch in enumerate(chunks):
            processed_chunks.append({
                "chunk_index": idx,
                "content": ch,
                "page_number": self.find_page(ch, pages_text),
                "chapter": "",
                "paragraph": "",
                "section_title": ""
            })

        return {
            "filename": filename,
            "total_pages": len(pages_text),
            "total_chunks": len(processed_chunks),
            "chunks": processed_chunks
        }

    # ---------------- PAGE MAPPING ----------------

    def find_page(self, chunk_text: str, pages_text: List[Dict]) -> int:
        for p in pages_text:
            if chunk_text[:100] in p["text"]:
                return p["page"]
        return 1
