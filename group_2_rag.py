import os
import re
import numpy as np
from dotenv import load_dotenv

from utils import (
    load_pdf,
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

def load_data_from_folder(folder_path):
    print(f"Scanning directory: {folder_path}")
    raw_docs_list = []
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return raw_docs_list

    for file in os.listdir(folder_path):
        if file.lower().endswith('.pdf'):
            full_path = os.path.join(folder_path, file)
            print(f"Loading document: {file}...")
            try:
                text_content = load_pdf(full_path)
                raw_docs_list.append({
                    "text": text_content,
                    "source": file
                })
            except Exception as e:
                print(f"Could not load {file}: {e}")
    return raw_docs_list
