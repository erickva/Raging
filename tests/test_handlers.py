import types
from pathlib import Path

import pytest

from raging.ingest.handlers import DocxHandler, ExcelHandler, PdfHandler, TextHandler
from raging.ingest.base import SourceDocument, to_source_id


class DummyDocument(SourceDocument):
    """Utility to instantiate SourceDocument from temp files."""


def _make_document(path: Path) -> SourceDocument:
    return SourceDocument(
        source_id=to_source_id(path),
        path=path,
        metadata={},
        tags=(),
    )


@pytest.fixture(autouse=True)
def clear_optional_modules():
    import sys

    modules = [
        "llama_index",
        "llama_index.readers",
        "llama_index.readers.file",
        "pypdf",
        "pandas",
        "docx",
        "langchain_text_splitters",
    ]
    backup = {name: sys.modules.get(name) for name in modules}
    for name in modules:
        sys.modules.pop(name, None)

    splitters = types.ModuleType("langchain_text_splitters")

    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size: int, chunk_overlap: int):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_text(self, text: str):
            return [text]

    class MarkdownHeaderTextSplitter:
        def __init__(self, headers_to_split_on):
            self.headers_to_split_on = headers_to_split_on

        def split_text(self, text: str):
            return [types.SimpleNamespace(page_content=text)]

    class QuestionAnswerSplitter:
        def split_text(self, text: str):
            qa_pairs = []
            for block in text.split("\n\n"):
                if "Q:" in block and "A:" in block:
                    question = block.split("A:")[0].split("Q:")[-1].strip()
                    answer = block.split("A:")[-1].strip()
                    qa_pairs.append(types.SimpleNamespace(question=question, answer=answer))
            return qa_pairs or [types.SimpleNamespace(question="", answer=text)]

    splitters.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
    splitters.MarkdownHeaderTextSplitter = MarkdownHeaderTextSplitter
    splitters.QuestionAnswerSplitter = QuestionAnswerSplitter
    sys.modules["langchain_text_splitters"] = splitters

    try:
        yield
    finally:
        sys.modules.update({k: v for k, v in backup.items() if v is not None})
        if backup.get("langchain_text_splitters") is None:
            sys.modules.pop("langchain_text_splitters", None)


def test_pdf_handler_uses_pypdf_fallback(monkeypatch, tmp_path: Path) -> None:
    fake_module = types.ModuleType("pypdf")

    class FakePage:
        def extract_text(self):
            return "Hello"

    class FakeReader:
        def __init__(self, path):
            self.pages = [FakePage(), FakePage()]

    fake_module.PdfReader = FakeReader
    monkeypatch.setitem(__import__("sys").modules, "pypdf", fake_module)

    path = tmp_path / "sample.pdf"
    path.write_text("dummy", encoding="utf-8")
    handler = PdfHandler()
    from raging.ingest.handlers import _extract_pdf_text

    text = _extract_pdf_text(path)
    assert text == "Hello\n\nHello"


def test_docx_handler_extracts_paragraphs(monkeypatch, tmp_path: Path) -> None:
    fake_docx = types.ModuleType("docx")

    class FakeParagraph:
        def __init__(self, text):
            self.text = text

    class FakeDoc:
        def __init__(self, path):
            self.paragraphs = [FakeParagraph("Para 1"), FakeParagraph(""), FakeParagraph("Para 2")]

    fake_docx.Document = FakeDoc
    monkeypatch.setitem(__import__("sys").modules, "docx", fake_docx)

    path = tmp_path / "sample.docx"
    path.write_text("", encoding="utf-8")
    handler = DocxHandler()
    from raging.ingest.handlers import _extract_docx_text

    text = _extract_docx_text(path)
    assert text == "Para 1\n\nPara 2"


def test_excel_handler_extracts_cells(monkeypatch, tmp_path: Path) -> None:
    fake_pandas = types.ModuleType("pandas")

    class FakeFrame:
        def __init__(self, rows):
            self._rows = rows
            self._columns = list(rows[0].keys()) if rows else []

        def fillna(self, value):
            return self

        @property
        def columns(self):
            return self._columns

        def to_numpy(self, dtype=str):
            return [[str(row[col]) for col in self._columns] for row in self._rows]

    def read_excel(path, sheet_name=None):
        return {"Sheet1": FakeFrame([{"A": 1, "B": "two"}])}

    fake_pandas.read_excel = read_excel
    monkeypatch.setitem(__import__("sys").modules, "pandas", fake_pandas)

    path = tmp_path / "sample.xlsx"
    path.write_text("", encoding="utf-8")
    handler = ExcelHandler()
    text = handler._extract_excel_text(path)
    assert "Sheet: Sheet1" in text
    assert "1" in text
    assert "two" in text


def test_text_handler_skips_blank_chunks(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("hello\n\n\nworld", encoding="utf-8")
    handler = TextHandler()
    document = _make_document(path)
    chunks = list(handler.iter_chunks(document, "sha256"))
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.content.strip()
