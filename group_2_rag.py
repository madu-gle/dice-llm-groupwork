import os
import re
from bisect import bisect_right
from collections import Counter

import numpy as np
import pypdf
from dotenv import load_dotenv

from utils import (
    build_bm25_index,
    bm25_search,
    load_embedding_model,
    embed_texts,
    semantic_search,
    create_llm_client,
)

# Load environment variables from .env
load_dotenv()

print("All imports loaded successfully.")

# ---------------------------------------------------------------------------
# Documents in the corpus
# ---------------------------------------------------------------------------

# Short name used in answers + the ways users refer to a document in a question.
# Files not listed here get their file name as short name.
DOCUMENT_ALIASES = {
    "eu_ai_act.pdf": ["AI Act", "AI-Act", "Artificial Intelligence Act", "2024/1689"],
    "CELEX_32016R0679_EN_TXT.pdf": ["GDPR", "General Data Protection Regulation", "2016/679"],
}


# ---------------------------------------------------------------------------
# PDF cleaning
# ---------------------------------------------------------------------------

# Page headers/footers of EUR-Lex PDFs that change on every page (page numbers)
FOOTER_PATTERNS = [
    re.compile(r"^(\d+/\d+ )?ELI: http"),                                 # AI Act: "ELI: http://data.europa.eu/eli/reg/2024/1689/oj 115/144", "88/144 ELI: http://..."
    re.compile(r"^\d{1,2}\.\d{1,2}\.\d{4} L \d+/\d+ Official Journal"),   # GDPR: "4.5.2016 L 119/2 Official Journal of the European Union EN"
    re.compile(r"^L \d+/\d+ \S+ Official Journal"),                      # GDPR, even pages
    re.compile(r"^EN$"),
]
FOOTNOTE_START = re.compile(r"^\((\d{1,3})\)\s")
FOOTNOTE_CITATION = re.compile(r"\bOJ [LC]\b|\bp\. \d+")
FOOTNOTE_MAX_LEN = 1200 # footnotes are short citations; anything longer is body text
# A line starting with one of these begins a new unit and must not be joined to the line before
NEW_UNIT = re.compile(r"^(\(\d+\)|\d+\.|\([a-z]+\)|\([ivx]+\)|Article \d+|CHAPTER|SECTION|Section \d|ANNEX|—)")
FULL_LINE = 60  # PDF lines wrapped mid-sentence are long; short lines are headings or paragraph ends


def remove_trailing_footnotes(lines: list[str]) -> list[str]:
    """Drops the footnotes at the bottom of a page.

    Walks up from the end of the page, grouping lines into items that start with "(n) ".
    An item is a footnote if
      - it is short (footnotes are citations, not body text),
      - its number is one less than the footnote below it (footnotes are consecutive), and
      - it cites the Official Journal ("OJ L ...", "p. 31") or its number is referenced
        inline earlier on the page ("... European Parliament (6).").
    Stops at the first item that fails, so recitals such as "(3) Directive 95/46/EC ..."
    directly above the footnotes are kept.
    """
    end = len(lines)
    expected = None  # number the next footnote (going up) must have
    i = end - 1
    while i >= 0:
        item = " ".join(lines[i:end])
        if len(item) > FOOTNOTE_MAX_LEN:
            break
        start = FOOTNOTE_START.match(lines[i])
        if start:
            number = int(start.group(1))
            body_above = " ".join(lines[:i])
            referenced = re.search(rf"\w[.,;:’)]? \({number}\)", body_above)
            consecutive = expected is None or number == expected
            if not (consecutive and (FOOTNOTE_CITATION.search(item) or referenced)):
                break
            end = i
            expected = number - 1
        i -= 1
    return lines[:end]


def clean_pages(pages: list[str]) -> list[str]:
    """Removes headers, footers and footnotes and repairs PDF line wrapping."""
    # Lines that repeat on many pages are running headers/footers ("OJ L, 12.7.2024 EN")
    counts = Counter(re.sub(r"\s+", " ", line).strip() for page in pages for line in set(page.splitlines()))
    boilerplate = {line for line, c in counts.items() if line and c > len(pages) * 0.3}

    cleaned = []
    for page in pages:
        lines = []
        for line in page.splitlines():
            line = re.sub(r"\s+", " ", line).strip()
            if not line or line in boilerplate or any(p.match(line) for p in FOOTER_PATTERNS):
                continue
            lines.append(line)
        lines = remove_trailing_footnotes(lines)

        # Re-join lines that were wrapped mid-sentence
        text = lines[0] if lines else ""
        for prev, line in zip(lines, lines[1:]):
            if len(prev) >= FULL_LINE and not NEW_UNIT.match(line):
                if prev.endswith("\u00ad"):    # soft hyphen: "pro\u00ad\ncessing" -> "processing"
                    text = text[:-1] + line
                elif prev.endswith("-"):        # real hyphen: "high-\nrisk" -> "high-risk"
                    text += line
                else:
                    text += " " + line
            else:
                text += "\n" + line
        cleaned.append(text.replace("\u00ad", ""))
    return cleaned


# ---------------------------------------------------------------------------
# PDF loading with metadata
# ---------------------------------------------------------------------------


def parse_pdf_date(value) -> str | None:
    """Turns a PDF date like "D:20240711144717+02'00'" into "2024-07-11"."""
    match = re.match(r"D:(\d{4})(\d{2})(\d{2})", str(value or ""))
    return "-".join(match.groups()) if match else None


def load_data_from_folder(folder_path, clean=True):
    """Loads every PDF in `folder_path`.

    Returns one dict per document:
        text        full (cleaned unless clean=False) text
        source      file name
        doc_name    short name, e.g. "AI Act" (see DOCUMENT_ALIASES)
        title       title from the PDF metadata
        created     creation date from the PDF metadata, e.g. "2024-07-11"
        page_starts character offset in `text` where each page starts
    """
    print(f"Scanning directory: {folder_path}")
    raw_docs_list = []
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return raw_docs_list

    for file in sorted(os.listdir(folder_path)):
        if file.lower().endswith('.pdf'):
            full_path = os.path.join(folder_path, file)
            print(f"Loading document: {file}...")
            try:
                reader = pypdf.PdfReader(full_path)
                pages = [page.extract_text() or "" for page in reader.pages]
                if clean:
                    pages = clean_pages(pages)

                page_starts, offset = [], 0
                for page in pages:
                    page_starts.append(offset)
                    offset += len(page) + 1  # +1 for the "\n" joining the pages

                meta = reader.metadata or {}
                title = " ".join(str(meta.get("/Title") or file).split()).replace("/ ", "/")
                raw_docs_list.append({
                    "text": "\n".join(pages),
                    "source": file,
                    "doc_name": DOCUMENT_ALIASES.get(file, [file])[0],
                    "title": title,
                    "created": parse_pdf_date(meta.get("/CreationDate")),
                    "page_starts": page_starts,
                })
            except Exception as e:
                print(f"Could not load {file}: {e}")
    return raw_docs_list


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

MARKER = re.compile(r"\(\d+\)")
SECTION_HEADING = re.compile(r"(?m)^(Article \d+|ANNEX [IVXLC]+)$")  # "Article 99", "ANNEX III"
RECITAL_START = re.compile(r"(?m)^\((\d+)\) ")
PREAMBLE_END = "HAVE ADOPTED THIS REGULATION"  # recitals come before this, articles after


def paragraph_aware_chunking(raw_docs, max_chunk_len=1000):
    """
    Splits each document at "(n)" markers once a chunk is longer than max_chunk_len.

    Returns (chunks, metadata): chunks is a list of strings, metadata a list of dicts
    (same order) with doc_name, source, title, created, page, recitals and sections.
    """
    chunks, metadata = [], []
    for doc in raw_docs:
        text = doc["text"]
        preamble_end = text.find(PREAMBLE_END)
        if preamble_end == -1:
            preamble_end = 0
        headings = [(m.start(), m.group(1)) for m in SECTION_HEADING.finditer(text)]
        recitals = [(m.start(), int(m.group(1))) for m in RECITAL_START.finditer(text[:preamble_end])]

        # chunk boundaries: same rule as before, but keeping character offsets
        bounds, start = [], 0
        for marker in MARKER.finditer(text):
            if marker.start() - start > max_chunk_len:
                bounds.append((start, marker.start()))
                start = marker.start()
        bounds.append((start, len(text)))

        for start, end in bounds:
            chunk = text[start:end].strip()
            if not chunk:
                continue
            chunks.append(chunk)
            metadata.append({
                "doc_name": doc.get("doc_name", doc["source"]),
                "source": doc["source"],
                "title": doc.get("title", doc["source"]),
                "created": doc.get("created"),
                "page": bisect_right(doc.get("page_starts", [0]), start),
                "recitals": items_in_range(recitals, start, end) if start < preamble_end else [],
                "sections": items_in_range(headings, start, end),
            })
    return chunks, metadata


def items_in_range(items, start, end):
    """Items (offset, value) that start inside [start, end), plus the one already running at `start`."""
    running = [value for offset, value in items if offset <= start]
    inside = [value for offset, value in items if start < offset < end]
    return running[-1:] + inside


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def detect_documents(query: str) -> set[str]:
    """Short names of the documents the query mentions ("GDPR", "AI Act"); empty = all."""
    def mentions(alias):
        return re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", query, re.I)
    return {aliases[0] for aliases in DOCUMENT_ALIASES.values() if any(mentions(a) for a in aliases)}


def parse_reference(query: str) -> tuple[str | None, int | None]:
    """Finds an explicit reference: ("Article 99", None), ("ANNEX III", None) or (None, 155) for a recital.

    "paragraph" only means a recital when no article is named ("paragraph 2 of Article 5" is not recital 2).
    """
    article = re.search(r"\b(?:article|art\.)\s*(\d+)", query, re.I)
    annex = re.search(r"\bannex\s+([IVXLC]+)\b", query, re.I)
    recital = re.search(r"\b(?:recital|paragraph)s?\s*\(?(\d+)\)?", query, re.I)
    section = f"Article {article.group(1)}" if article else f"ANNEX {annex.group(1).upper()}" if annex else None
    if recital and section and not re.search(r"\brecital", query, re.I):
        recital = None
    return section, int(recital.group(1)) if recital else None


def retrieve(
    query: str,
    bm25_index: object,
    embed_model: object,
    corpus_embeddings: np.ndarray,
    chunks: list[str],
    chunk_metadata: list[dict],
    top_k: int = 5,
    max_exact: int = 3,
) -> list[int]:
    """Metadata-aware hybrid retrieval. Returns the indices of the chunks passed to the LLM.

    1. Only chunks of the documents named in the query are considered (all if none is named).
    2. Chunks matching an explicit reference ("Article 99", "paragraph 155") come first.
    3. The rest is filled with BM25 + semantic search results for the full query.
    """
    docs = detect_documents(query)
    allowed = {i for i, m in enumerate(chunk_metadata) if not docs or m["doc_name"] in docs}

    section, recital = parse_reference(query)
    exact = [
        i for i in sorted(allowed)
        if (section and section in chunk_metadata[i]["sections"])
        or (recital and recital in chunk_metadata[i]["recitals"])
    ][:max_exact]

    bm25_results = bm25_search(query, bm25_index, chunks, top_k=len(chunks))
    semantic_results = semantic_search(query, embed_model, corpus_embeddings, chunks, top_k=len(chunks))
    bm25_ranked = [r["chunk_idx"] for r in bm25_results if r["chunk_idx"] in allowed][:top_k]
    semantic_ranked = [r["chunk_idx"] for r in semantic_results if r["chunk_idx"] in allowed][:top_k]

    results = []
    for idx in exact + bm25_ranked + semantic_ranked:
        if idx not in results:
            results.append(idx)
    return results[:max(top_k, len(exact))]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def chunk_label(meta: dict) -> str:
    """Human-readable reference for a chunk, e.g. "AI Act, Article 99, p. 115"."""
    parts = [meta["doc_name"]]
    if meta["recitals"]:
        parts.append("Recital " + ", ".join(map(str, meta["recitals"][:3])) + ("..." if len(meta["recitals"]) > 3 else ""))
    if meta["sections"]:
        parts.append(", ".join(meta["sections"][:3]))
    parts.append(f"p. {meta['page']}")
    return ", ".join(parts)


def rag_generate(
    query: str,
    llm_client: object,
    bm25_index: object,
    embed_model: object,
    corpus_embeddings: np.ndarray,
    chunks: list[str],
    chunk_metadata: list[dict],
    top_k: int = 5,
    return_indices: bool = False,
):
    relevant_indices = retrieve(query, bm25_index, embed_model, corpus_embeddings, chunks, chunk_metadata, top_k=top_k)

    # Document info (title, creation date) once per document, then each chunk with its label
    doc_info = {}
    for idx in relevant_indices:
        m = chunk_metadata[idx]
        doc_info[m["doc_name"]] = f"- {m['doc_name']}: {m['title']} (PDF created {m['created'] or 'unknown'})"
    context_blocks = [f"[{chunk_label(chunk_metadata[idx])}]\n{chunks[idx]}" for idx in relevant_indices]
    focused_context = "Documents:\n" + "\n".join(doc_info.values()) + "\n\n" + "\n\n".join(context_blocks)

    system_prompt = (
        "You are a helpful assistant for EU regulations. Answer the question using only the provided context. "
        "Each context passage starts with its source in square brackets; cite the document and article or recital "
        "you used, e.g. [AI Act, Article 99]. If a paragraph is requested and the paragraph is not in the context, "
        "state that it is missing."
    )
    final_user_prompt = f"Context:\n{focused_context}\n\nQuestion: {query}"

    response = llm_client.generate(history=[{"role": "system", "content": system_prompt}], prompt=final_user_prompt, temperature=1.0, max_tokens=1024)

    sources = list(dict.fromkeys(chunk_label(chunk_metadata[idx]) for idx in relevant_indices))
    if return_indices:
        return response, sources, relevant_indices
    return response, sources
