"""PDF loader — one Document per page.

Why per-page (not one Document for the whole PDF): page_number is valuable
citation metadata ("see page 12"), and it keeps individual Documents small
enough for the chunker to work with cleanly.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

from pypdf import PdfReader

from ..schema import Document, DocumentMetadata
from .base import BaseLoader


class PDFLoader(BaseLoader):
    doc_type = "pdf"

    def load(self, file_path: Union[str, Path]) -> List[Document]:
        file_path = Path(file_path)
        reader = PdfReader(str(file_path))

        info = reader.metadata or {}
        title = info.get("/Title") or file_path.stem
        author = info.get("/Author")
        created_date = info.get("/CreationDate")

        documents: List[Document] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue  # skip blank/image-only pages
            documents.append(
                Document(
                    content=text,
                    metadata=DocumentMetadata(
                        source=str(file_path),
                        doc_type=self.doc_type,
                        title=title,
                        author=author,
                        created_date=created_date,
                        page_number=page_number,
                    ),
                )
            )
        return documents
