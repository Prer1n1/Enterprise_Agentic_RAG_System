"""Loader registry: pick the right format-specific loader by file extension."""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

from ..schema import Document
from .base import BaseLoader
from .csv_loader import CSVLoader
from .docx_loader import DOCXLoader
from .html_loader import HTMLLoader
from .pdf_loader import PDFLoader

_LOADER_MAP: dict[str, type[BaseLoader]] = {
    ".pdf": PDFLoader,
    ".docx": DOCXLoader,
    ".html": HTMLLoader,
    ".htm": HTMLLoader,
    ".csv": CSVLoader,
}


def get_loader(file_path: Union[str, Path]) -> BaseLoader:
    ext = Path(file_path).suffix.lower()
    loader_cls = _LOADER_MAP.get(ext)
    if loader_cls is None:
        raise ValueError(
            f"No loader registered for file extension '{ext}'. "
            f"Supported: {sorted(_LOADER_MAP)}"
        )
    return loader_cls()


def load_document(file_path: Union[str, Path]) -> List[Document]:
    """Entry point the rest of the pipeline calls — format is invisible
    to callers past this function."""
    return get_loader(file_path).load(file_path)


__all__ = [
    "BaseLoader",
    "PDFLoader",
    "DOCXLoader",
    "HTMLLoader",
    "CSVLoader",
    "get_loader",
    "load_document",
]
