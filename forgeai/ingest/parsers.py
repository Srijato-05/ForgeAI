"""Document parsers and folder inventory builder for ForgeAI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


class CSVProfile(BaseModel):
    """Schema and sample preview of a CSV file (never embedded as text chunks)."""

    filename: str
    path: str
    columns: list[str]
    dtypes: dict[str, str]
    row_count: int
    sample_rows: list[dict[str, Any]] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    """Normalized text extracted from a non-tabular document."""

    source_path: str
    filename: str
    file_type: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentInventory(BaseModel):
    """High-level summary of all documents and CSV schemas in a folder."""

    documents: list[dict[str, Any]] = Field(default_factory=list)
    csv_profiles: list[CSVProfile] = Field(default_factory=list)


def _extract_pdf_text(path: Path) -> str:
    try:
        import pymupdf4llm  # type: ignore

        return str(pymupdf4llm.to_markdown(str(path)))
    except Exception:
        import pypdf

        reader = pypdf.PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)


def _extract_docx_text(path: Path) -> str:
    import docx

    doc = docx.Document(str(path))
    parts: list[str] = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def _profile_csv(path: Path) -> CSVProfile:
    df = pd.read_csv(path)
    sample = df.head(3).fillna("").to_dict(orient="records")
    dtypes = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
    return CSVProfile(
        filename=path.name,
        path=str(path.resolve()),
        columns=[str(c) for c in df.columns],
        dtypes=dtypes,
        row_count=int(len(df)),
        sample_rows=sample,
    )


def parse_folder(docs_dir: Path) -> tuple[list[ParsedDocument], DocumentInventory]:
    """Parse all supported documents in ``docs_dir`` and build a DocumentInventory."""
    root = Path(docs_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Document folder not found: {root}")

    parsed_docs: list[ParsedDocument] = []
    inventory_docs: list[dict[str, Any]] = []
    csv_profiles: list[CSVProfile] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        rel_name = path.relative_to(root).as_posix()

        if suffix == ".csv":
            profile = _profile_csv(path)
            csv_profiles.append(profile)
            continue

        text = ""
        file_type = suffix.lstrip(".")
        if suffix == ".pdf":
            text = _extract_pdf_text(path)
        elif suffix in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="replace")
        elif suffix == ".docx":
            text = _extract_docx_text(path)
        else:
            continue

        cleaned = text.strip()
        if not cleaned:
            continue

        parsed_docs.append(
            ParsedDocument(
                source_path=str(path.resolve()),
                filename=rel_name,
                file_type=file_type,
                text=cleaned,
                metadata={"char_count": len(cleaned), "word_count": len(cleaned.split())},
            )
        )
        preview = " ".join(cleaned.split()[:40])
        inventory_docs.append(
            {
                "filename": rel_name,
                "file_type": file_type,
                "word_count": len(cleaned.split()),
                "preview": preview,
            }
        )

    inventory = DocumentInventory(documents=inventory_docs, csv_profiles=csv_profiles)
    return parsed_docs, inventory

