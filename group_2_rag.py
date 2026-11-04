import os
import re
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

# TODOs
# file creation date not working, but works when single doc is given

def paragraph_aware_chunking(raw_docs, max_chunk_len=1000):
    """
    Accepts raw_docs: list of dicts with 'text' and 'source' or a single string.
    Returns (all_chunks, all_sources) where each chunk is aligned with its source file.
    """
    normalized_docs = []
    if isinstance(raw_docs, str):
        normalized_docs.append({"text": raw_docs, "source": "unknown_source"})
    elif isinstance(raw_docs, list):
        for item in raw_docs:
            if isinstance(item, dict):
                normalized_docs.append({
                    "text": item.get("text", ""),
                    "source": item.get("source", "unknown_source")
                })
            elif isinstance(item, str):
                normalized_docs.append({"text": item, "source": "unknown_source"})

    marker_pattern = r'(\(\d+\))'
    all_chunks = []
    all_sources = []

    for doc in normalized_docs:
        raw_text = doc["text"]
        source_name = doc["source"]
        parts = re.split(marker_pattern, raw_text)

        current_chunk = ""
        i = 0
        while i < len(parts):
            segment = parts[i]
            if re.match(marker_pattern, segment):
                if len(current_chunk) > max_chunk_len and current_chunk:
                    all_chunks.append(current_chunk.strip())
                    all_sources.append(source_name)
                    current_chunk = ""
                current_chunk += "\n" + segment
            else:
                current_chunk += segment
            i += 1

        if current_chunk:
            all_chunks.append(current_chunk.strip())
            all_sources.append(source_name)

    return all_chunks, all_sources

def retrieve(
    query: str,
    bm25_index: object,
    embed_model: object,
    corpus_embeddings: np.ndarray,
    chunks: list[str],
    top_k: int = 5,
) -> list[int]:
    """Hybrid BM25 + semantic retrieval with paragraph-number filtering.

    Returns the indices of the chunks that are passed to the LLM as context.
    Split out of `rag_generate` so the evaluation script can score retrieval.
    """
    match = re.search(r'\(?(\d+)\)?', query)
    if match:
        paragraph_num = match.group(1)
        lookup_key = f"({paragraph_num})"
        prev_paragraph = f"({int(paragraph_num)-1})"
    else:
        lookup_key = query
        prev_paragraph = ""

    bm25_results = bm25_search(lookup_key, bm25_index, chunks, top_k=top_k)
    semantic_results = semantic_search(lookup_key, embed_model, corpus_embeddings, chunks, top_k=top_k)

    retrieved_indices = []
    seen = set()
    for result in bm25_results + semantic_results:
        idx = result["chunk_idx"]
        if idx not in seen:
            seen.add(idx)
            retrieved_indices.append(idx)

    relevant_indices = []
    for idx in retrieved_indices:
        chunk_text = chunks[idx]
        if lookup_key in chunk_text or (prev_paragraph and prev_paragraph in chunk_text):
            relevant_indices.append(idx)

    if not relevant_indices:
        relevant_indices = retrieved_indices[:2]

    return relevant_indices


def rag_generate(
    query: str,
    llm_client: object,
    bm25_index: object,
    embed_model: object,
    corpus_embeddings: np.ndarray,
    chunks: list[str],
    chunk_sources: list[str],
    top_k: int = 5,
    return_indices: bool = False,
):
    relevant_indices = retrieve(query, bm25_index, embed_model, corpus_embeddings, chunks, top_k=top_k)

    focused_chunks = [chunks[idx] for idx in relevant_indices]
    sources = [chunk_sources[idx] for idx in relevant_indices]
    focused_context = "\n\n".join(focused_chunks)

    system_prompt = "You are a helpful assistant. Answer the question using only the provided context. If a paragraph is requested and the paragraph is not in the context, state that it is missing."
    final_user_prompt = f"Context:\n{focused_context}\n\nQuestion: {query}"

    response = llm_client.generate(history=[{"role": "system", "content": system_prompt}], prompt=final_user_prompt, temperature=1.0, max_tokens=1024)

    if return_indices:
        return response, list(set(sources)), relevant_indices
    return response, list(set(sources))

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


def load_data_from_folder(folder_path, clean=True):
    """Loads every PDF in `folder_path` as {"text", "source"}; cleans the text unless clean=False."""
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
                pages = [page.extract_text() or "" for page in pypdf.PdfReader(full_path).pages]
                if clean:
                    pages = clean_pages(pages)
                raw_docs_list.append({
                    "text": "\n".join(pages),
                    "source": file
                })
            except Exception as e:
                print(f"Could not load {file}: {e}")
    return raw_docs_list
