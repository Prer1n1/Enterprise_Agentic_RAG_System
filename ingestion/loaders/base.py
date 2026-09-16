"""Loader interface every format-specific loader implements.

Why an abstract base at all: it forces every loader (PDF/DOCX/HTML/CSV) to
return the same shape (List[Document]), which is what lets the rest of the
pipeline treat all four formats identically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

from ..schema import Document


class BaseLoader(ABC):
    doc_type: str

    @abstractmethod
    def load(self, file_path: Union[str, Path]) -> List[Document]:
        """Load one file and return it as a list of normalized Documents."""
        raise NotImplementedError
