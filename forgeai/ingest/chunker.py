"""Heading-aware document chunker with contextual prefix enrichment."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from forgeai.ingest.parsers import ParsedDocument

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)


class Chunk(BaseModel):
    """A single retrievable text chunk with contextual enrichment."""

    chunk_id: str
    source_doc: str
    section_title: str
    raw_text: str
    contextual_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("General", text.strip())]

    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("Introduction", preamble))

    for idx, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((title, body))
    return sections or [("General", text.strip())]


def _window_words(text: str, max_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]

    step = max(1, max_words - overlap_words)
    windows: list[str] = []
    for start in range(0, len(words), step):
        slice_words = words[start : start + max_words]
        windows.append(" ".join(slice_words))
        if start + max_words >= len(words):
            break
    return windows


def chunk_documents(
    docs: list[ParsedDocument],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    contextualizer: Callable[[ParsedDocument, str, str], str] | None = None,
) -> list[Chunk]:
    """Split parsed documents by Markdown headings then sliding word windows."""
    max_words = max(20, int(chunk_size / 1.3))
    overlap_words = min(max_words - 1, max(0, int(chunk_overlap / 1.3)))

    chunks: list[Chunk] = []
    for doc in docs:
        stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(doc.filename).stem).strip("_") or "doc"
        doc_idx = 0
        for section_title, section_body in _split_by_headings(doc.text):
            for raw in _window_words(section_body, max_words=max_words, overlap_words=overlap_words):
                prefix = (
                    contextualizer(doc, section_title, raw)
                    if contextualizer is not None
                    else f"[Document: {doc.filename} | Section: {section_title}]"
                )
                contextual_text = f"{prefix}\n{raw}".strip()
                chunk_id = f"{stem}__c{doc_idx:03d}"
                doc_idx += 1
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        source_doc=doc.filename,
                        section_title=section_title,
                        raw_text=raw,
                        contextual_text=contextual_text,
                        metadata={"file_type": doc.file_type, "word_count": len(raw.split())},
                    )
                )
    return chunks

