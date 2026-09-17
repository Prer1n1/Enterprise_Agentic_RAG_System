"""CSV loader — one Document per row.

Why per-row (not one Document for the whole file): CSV is already
structured data, so the natural "unit of meaning" is a record, not an
arbitrary text span. This also sets up row-based chunking downstream —
the chunker can batch these single-row Documents without ever needing to
understand CSV itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import pandas as pd

from ..schema import Document, DocumentMetadata, canonical_source
from .base import BaseLoader

# If a CSV has one of these columns, its value is ground truth for category
# classification — a "department" column saying "IT" is authoritative and
# should NOT be overridden by keyword-matching the row's free text (e.g. a
# row about "password rotation" tagged department=IT would otherwise get
# misclassified as "Security" just because "password" is a Security keyword).
_CATEGORY_HINT_COLUMNS = {"department", "category"}


class CSVLoader(BaseLoader):
    doc_type = "csv"

    def load(self, file_path: Union[str, Path]) -> List[Document]:
        file_path = Path(file_path)
        df = pd.read_csv(file_path)
        columns = list(df.columns)
        hint_column = next((c for c in columns if c.strip().lower() in _CATEGORY_HINT_COLUMNS), None)

        documents: List[Document] = []
        for row_index, row in df.iterrows():
            content = "\n".join(f"{col}: {row[col]}" for col in columns)
            extra = {"row_index": int(row_index), "columns": columns}
            if hint_column:
                extra["category_hint"] = str(row[hint_column]).strip()
            documents.append(
                Document(
                    content=content,
                    metadata=DocumentMetadata(
                        source=canonical_source(file_path),
                        doc_type=self.doc_type,
                        title=file_path.stem,
                        section=f"Row {row_index + 1}",
                        extra=extra,
                    ),
                )
            )
        return documents
