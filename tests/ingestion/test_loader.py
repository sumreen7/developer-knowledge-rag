import pytest
from langchain_core.documents import Document
from src.ingestion.loader import DocumentLoader


@pytest.fixture
def temp_docs_dir(tmp_path):
    (tmp_path / "auth.md").write_text("# Auth\n\nUse Bearer tokens.")
    (tmp_path / "guide.txt").write_text("Step 1: install the SDK.")
    (tmp_path / "config.json").write_text('{"key": "value"}')
    return tmp_path


def test_loader_initializes(temp_docs_dir):
    loader = DocumentLoader(str(temp_docs_dir))
    assert loader.data_directory == temp_docs_dir


def test_missing_directory_raises():
    with pytest.raises(ValueError, match="does not exist"):
        DocumentLoader("/nonexistent/path")


def test_loads_markdown(temp_docs_dir):
    loader = DocumentLoader(str(temp_docs_dir))
    docs = loader.load_file(temp_docs_dir / "auth.md")
    assert len(docs) >= 1
    assert isinstance(docs[0], Document)


def test_metadata_fields_present(temp_docs_dir):
    loader = DocumentLoader(str(temp_docs_dir))
    docs = loader.load_file(temp_docs_dir / "auth.md")
    meta = docs[0].metadata
    assert "source" in meta
    assert "file_name" in meta
    assert "file_type" in meta
    assert "load_timestamp" in meta


def test_unsupported_file_skipped(temp_docs_dir):
    loader = DocumentLoader(str(temp_docs_dir))
    docs = loader.load_file(temp_docs_dir / "config.json")
    assert docs == []


def test_load_directory(temp_docs_dir):
    loader = DocumentLoader(str(temp_docs_dir))
    docs = loader.load_directory()
    assert len(docs) >= 2


def test_empty_directory(tmp_path):
    loader = DocumentLoader(str(tmp_path))
    docs = loader.load_directory()
    assert docs == []
