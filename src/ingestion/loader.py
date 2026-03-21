import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
)

logger = logging.getLogger(__name__)

LOADER_MAPPING = {
    ".md":  UnstructuredMarkdownLoader,
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
}


class DocumentLoadError(Exception):
    pass


class DocumentLoader:
    def __init__(self, data_directory: str):
        self.data_directory = Path(data_directory)
        if not self.data_directory.exists():
            raise ValueError(
                f"Data directory does not exist: {self.data_directory}"
            )
        logger.info(f"DocumentLoader ready: {self.data_directory}")

    def load_file(self, file_path: Path) -> List[Document]:
        extension = file_path.suffix.lower()

        if extension not in LOADER_MAPPING:
            logger.warning(f"Skipping unsupported file: {file_path.name}")
            return []

        try:
            loader = LOADER_MAPPING[extension](str(file_path))
            raw_docs = loader.load()

            documents = []
            for doc in raw_docs:
                documents.append(Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "source":         str(file_path),
                        "file_name":      file_path.name,
                        "file_type":      extension.lstrip("."),
                        "file_stem":      file_path.stem,
                        "load_timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                ))

            logger.info(f"Loaded {len(documents)} doc(s) from {file_path.name}")
            return documents

        except Exception as e:
            raise DocumentLoadError(
                f"Failed to load {file_path.name}: {e}"
            ) from e

    def load_directory(
        self,
        recursive: bool = True,
        skip_errors: bool = True
    ) -> List[Document]:

        pattern = "**/*" if recursive else "*"
        all_files = [
            f for f in self.data_directory.glob(pattern) if f.is_file()
        ]

        if not all_files:
            logger.warning(f"No files found in {self.data_directory}")
            return []

        logger.info(f"Found {len(all_files)} files to process")

        all_documents = []
        failed = []

        for file_path in sorted(all_files):
            try:
                docs = self.load_file(file_path)
                all_documents.extend(docs)
            except DocumentLoadError as e:
                failed.append(file_path.name)
                if skip_errors:
                    logger.error(str(e))
                else:
                    raise

        logger.info(
            f"Done: {len(all_documents)} documents loaded, "
            f"{len(failed)} failed"
        )
        return all_documents

    def get_supported_extensions(self) -> List[str]:
        return list(LOADER_MAPPING.keys())