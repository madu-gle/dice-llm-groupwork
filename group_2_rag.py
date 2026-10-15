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

def rag_generate(
    query: str,
    llm_client: object,
    bm25_index: object,
    embed_model: object,
    corpus_embeddings: np.ndarray,
    chunks: list[str],
    chunk_sources: list[str],
    top_k: int = 5,
) -> tuple[str, list[str]]:
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
    seen_contents = set()

    for result in bm25_results + semantic_results:
        if isinstance(result, dict) and 'content' in result:
            content = result['content']
            if content not in seen_contents:
                seen_contents.add(content)
                if content in chunks:
                    retrieved_indices.append(chunks.index(content))

    relevant_indices = []
    for idx in retrieved_indices:
        chunk_text = chunks[idx]
        if lookup_key in chunk_text or (prev_paragraph and prev_paragraph in chunk_text):
            relevant_indices.append(idx)

    if not relevant_indices:
         relevant_indices = retrieved_indices[:2]

    focused_chunks = [chunks[idx] for idx in relevant_indices]
    sources = [chunk_sources[idx] for idx in relevant_indices]
    focused_context = "\n\n".join(focused_chunks)

    system_prompt = "You are a helpful assistant. Answer the question using only the provided context. If a paragraph is requested and the paragraph is not in the context, state that it is missing."
    final_user_prompt = f"Context:\n{focused_context}\n\nQuestion: {query}"

    response = llm_client.generate(history=[{"role": "system", "content": system_prompt}], prompt=final_user_prompt, temperature=1.0, max_tokens=1024)

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

def prepare_index(documents):
    print("Building BM25 and Semantic search indices...")
    bm25_index = build_bm25_index(documents)
    embedding_model = load_embedding_model()
    doc_embeddings = embed_texts(documents, embedding_model)
    return bm25_index, embedding_model, doc_embeddings

def main():
    load_dotenv()
    folder_path = "data"
    raw_docs = load_data_from_folder(folder_path)
    if not raw_docs:
        print("No documents loaded. Exiting.")
        return

    print("Applying paragraph-aware chunking...")
    documents, chunk_sources = paragraph_aware_chunking(raw_docs, max_chunk_len=1000)
    print(f"Created {len(documents)} paragraph-preserved chunks from loaded documents.")

    bm25_index, embedding_model, doc_embeddings = prepare_index(documents)

    llm_client = create_llm_client()

    # call for retrieval & llm inference
    # call for retrieval & llm inference
    prompts = [
        "What is written in paragraph 155?",
        "Who is the King of Sweden?",
        "What is the document about? Answer only in two sentences.",
        "What is the EU AI Act documents creation date?"
    ]

    for user_prompt in prompts:
        print(f"Running RAG pipeline for prompt: '{user_prompt}'")

        answer, sources = rag_generate(
            query=user_prompt,
            llm_client=llm_client,
            bm25_index=bm25_index,
            embed_model=embedding_model,
            corpus_embeddings=doc_embeddings,
            chunks=documents,
            chunk_sources=chunk_sources,
            top_k=10
        )

        print("\n=== RAG Response ===")
        print(answer)
        print("\n=== Sources ===")
        print(sources)



if __name__ == "__main__":
    main()
