"""Built-in ingestion handlers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

from .base import (
    DocumentChunk,
    IngestionHandler,
    SourceDocument,
    build_chunk_id,
    compute_checksum,
    iter_chunk_indices,
)
from .registry import get_registry


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class TextHandler(IngestionHandler):
    """Plain-text ingestion using a recursive character splitter."""

    name = "text"

    chunk_size: int = 800
    chunk_overlap: int = 200

    def _split(self, text: str) -> List[str]:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "langchain-text-splitters is required for TextHandler; install raging[ingest]"
            ) from exc

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        return splitter.split_text(text)

    def iter_chunks(self, document: SourceDocument, checksum_algorithm: str):
        text = _read_text(document.path)
        yield from self._emit_chunks(text, document, checksum_algorithm)

    # ------------------------------------------------------------------
    def _emit_chunks(
        self,
        text: str,
        document: SourceDocument,
        checksum_algorithm: str,
    ):
        for index, content in zip(iter_chunk_indices(), self._split(text)):
            if not content.strip():
                continue
            checksum = compute_checksum(content, algorithm=checksum_algorithm)
            metadata = {
                "source_path": str(document.path),
                "handler": self.name,
                **document.metadata,
            }
            yield DocumentChunk(
                chunk_id=build_chunk_id(document.source_id, index),
                source_id=document.source_id,
                content=content,
                checksum=checksum,
                chunk_index=index,
                metadata=metadata,
                tags=document.tags,
            )


class MarkdownHandler(TextHandler):
    """Markdown ingestion that respects header structure."""

    name = "markdown"
    min_header_level: int = 1
    max_header_level: int = 3

    def _split(self, text: str) -> List[str]:
        try:
            from langchain_text_splitters import MarkdownHeaderTextSplitter
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "langchain-text-splitters is required for MarkdownHandler; install raging[ingest]"
            ) from exc

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                (f"#{'#' * level}", f"header_level_{level}")
                for level in range(self.min_header_level, self.max_header_level + 1)
            ]
        )

        documents = splitter.split_text(text)
        return [doc.page_content for doc in documents]


class FAQHandler(TextHandler):
    """Ingestion for FAQ-style documents (Q/A pairs)."""

    name = "faq"

    def _split(self, text: str) -> List[str]:
        try:
            from langchain_text_splitters import QuestionAnswerSplitter
        except ImportError:  # pragma: no cover - optional dependency
            splitter = None
        else:
            qa_splitter = QuestionAnswerSplitter()
            return [f"Q: {qa.question}\nA: {qa.answer}" for qa in qa_splitter.split_text(text)]

        # Fallback: naive regex-based split on Q:/A:
        lines = text.splitlines()
        chunks: List[str] = []
        buffer: List[str] = []
        for line in lines:
            if line.strip().lower().startswith("q:") and buffer:
                chunks.append("\n".join(buffer).strip())
                buffer = [line]
            else:
                buffer.append(line)
        if buffer:
            chunks.append("\n".join(buffer).strip())
        return chunks


class PlaceholderHandler(IngestionHandler):
    """Placeholder for handlers that need future implementation."""

    name = "placeholder"

    def __init__(self, feature_name: str) -> None:
        self.feature_name = feature_name

    def iter_chunks(self, document: SourceDocument, checksum_algorithm: str):  # pragma: no cover
        raise NotImplementedError(
            f"Handler for {self.feature_name} is not yet implemented. "
            "Contribute via raging.plugins or install a project-specific handler."
        )


class PdfHandler(TextHandler):
    name = "pdf"

    def iter_chunks(self, document: SourceDocument, checksum_algorithm: str):
        text = self._extract_pdf_text(document.path)
        if not text:
            return
        yield from self._emit_chunks(text, document, checksum_algorithm)

    def _extract_pdf_text(self, path: Path) -> str:
        try:
            from llama_index.readers.file import PDFReader  # type: ignore
        except ImportError:
            PDFReader = None

        if PDFReader is not None:
            reader = PDFReader()
            documents = reader.load_data(file=path)
            return "\n\n".join(doc.text for doc in documents if getattr(doc, "text", "")).strip()

        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "PDF ingestion requires llama-index-readers-file or pypdf; install raging[pdf]."
            ) from exc

        pdf_reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in pdf_reader.pages]
        return "\n\n".join(pages).strip()


class ExcelHandler(TextHandler):
    name = "excel"

    def iter_chunks(self, document: SourceDocument, checksum_algorithm: str):
        text = self._extract_excel_text(document.path)
        if not text:
            return
        yield from self._emit_chunks(text, document, checksum_algorithm)

    def _extract_excel_text(self, path: Path) -> str:
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Excel ingestion requires pandas; install raging[excel].") from exc

        sheets = pd.read_excel(path, sheet_name=None)

        parts = []
        for sheet_name, frame in sheets.items():
            display = frame.fillna("")
            columns = [str(col) for col in display.columns]
            rows = [
                " | ".join(str(value) for value in row)
                for row in display.to_numpy(dtype=str)
            ]
            section_lines = [f"Sheet: {sheet_name}"]
            if columns:
                section_lines.append("Columns: " + " | ".join(columns))
            section_lines.extend(rows)
            parts.append("\n".join(section_lines))
        return "\n\n".join(parts).strip()


class DocxHandler(TextHandler):
    name = "docx"

    def iter_chunks(self, document: SourceDocument, checksum_algorithm: str):
        text = self._extract_docx_text(document.path)
        if not text:
            return
        yield from self._emit_chunks(text, document, checksum_algorithm)

    def _extract_docx_text(self, path: Path) -> str:
        try:
            import docx  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("DOCX ingestion requires python-docx; install raging[docx].") from exc

        document = docx.Document(str(path))
        paragraphs = [para.text for para in document.paragraphs if para.text.strip()]
        return "\n\n".join(paragraphs).strip()


DEFAULT_HANDLER_PATTERNS = {
    "markdown": ["*.md", "*.markdown"],
    "text": ["*.txt", "*.text"],
    "faq": ["*faq*.txt", "*faq*.md"],
    "pdf": ["*.pdf"],
    "excel": ["*.xls", "*.xlsx"],
    "docx": ["*.docx"],
}


def _safe_register(registry, name, handler, patterns):
    try:
        registry.register(name, handler, patterns=patterns)
    except ValueError:  # handler already registered
        pass


def register_default_handlers() -> None:
    registry = get_registry()
    _safe_register(registry, "text", TextHandler, DEFAULT_HANDLER_PATTERNS["text"])
    _safe_register(registry, "markdown", MarkdownHandler, DEFAULT_HANDLER_PATTERNS["markdown"])
    _safe_register(registry, "faq", FAQHandler, DEFAULT_HANDLER_PATTERNS["faq"])
    _safe_register(registry, "pdf", PdfHandler, DEFAULT_HANDLER_PATTERNS["pdf"])
    _safe_register(registry, "excel", ExcelHandler, DEFAULT_HANDLER_PATTERNS["excel"])
    _safe_register(registry, "docx", DocxHandler, DEFAULT_HANDLER_PATTERNS["docx"])


register_default_handlers()

__all__ = [
    "TextHandler",
    "MarkdownHandler",
    "FAQHandler",
    "PdfHandler",
    "ExcelHandler",
    "DocxHandler",
    "register_default_handlers",
]
