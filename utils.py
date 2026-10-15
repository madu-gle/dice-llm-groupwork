"""
utils.py — Utility module for the Document Assistant Final Project.

This module provides ready-to-use helper functions for:
  - Loading and parsing PDF documents.
  - Building and querying BM25 keyword search indices.
  - Loading embedding models and performing semantic (vector) search.
  - Interacting with LLMs via a unified LLMClient interface that supports
    OpenAI API, Azure OpenAI, and local Hugging Face transformers models.

These functions are provided as a STARTING POINT. You are free to modify,
extend, or replace any of them to fit your project's needs.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np
import pypdf
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# PDF Loading
# ---------------------------------------------------------------------------


def load_pdf(file_path: str) -> str:
    """
    Extracts and returns the full text content of a PDF file.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        A single string containing the concatenated text of all pages.
    """
    reader = pypdf.PdfReader(file_path)
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


# ---------------------------------------------------------------------------
# BM25 Keyword Search
# ---------------------------------------------------------------------------


def build_bm25_index(chunks: list[str]) -> BM25Okapi:
    """
    Tokenizes a list of text chunks and builds a BM25 index.

    Args:
        chunks: List of text strings (your corpus chunks).

    Returns:
        A BM25Okapi index object ready for querying.
    """
    tokenized = [chunk.lower().split() for chunk in chunks]
    return BM25Okapi(tokenized)


def bm25_search(
    query: str,
    index: BM25Okapi,
    chunks: list[str],
    top_k: int = 5,
) -> list[dict]:
    """
    Searches a BM25 index and returns the top-k results.

    Args:
        query:  The search query string.
        index:  A BM25Okapi index (built with `build_bm25_index`).
        chunks: The original list of text chunks (same order as the index).
        top_k:  Number of results to return.

    Returns:
        A list of dicts, each with keys: "chunk_idx", "content", "score".
        Results are sorted by descending BM25 score.
    """
    tokenized_query = query.lower().split()
    scores = index.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {"chunk_idx": int(i), "content": chunks[i], "score": float(scores[i])}
        for i in top_indices
    ]


# ---------------------------------------------------------------------------
# Semantic (Vector) Search
# ---------------------------------------------------------------------------


def load_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> SentenceTransformer:
    """
    Loads and returns a SentenceTransformer embedding model.

    Args:
        model_name: Hugging Face model identifier. The default is a small,
            fast model suitable for CPU inference.

    Returns:
        A SentenceTransformer model instance.
    """
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """
    Embeds a list of texts using the provided SentenceTransformer model.

    Use this function both for pre-computing corpus embeddings (once, offline)
    and for embedding a single query on the fly.

    Args:
        texts: List of strings to embed.
        model: A SentenceTransformer model instance.

    Returns:
        A numpy array of shape (len(texts), embedding_dim).
    """
    return model.encode(texts, show_progress_bar=len(texts) > 10)


def semantic_search(
    query: str,
    model: SentenceTransformer,
    corpus_embeddings: np.ndarray,
    chunks: list[str],
    top_k: int = 5,
) -> list[dict]:
    """
    Embeds the query on the fly, computes cosine similarity against
    pre-computed corpus embeddings, and returns the top-k results.

    Args:
        query:             The search query string.
        model:             A SentenceTransformer model (same one used to
                           produce `corpus_embeddings`).
        corpus_embeddings: Pre-computed embeddings for all chunks, shape
                           (n_chunks, embedding_dim).
        chunks:            The original list of text chunks.
        top_k:             Number of results to return.

    Returns:
        A list of dicts, each with keys: "chunk_idx", "content", "score".
        Results are sorted by descending cosine similarity.
    """
    query_embedding = model.encode(query)

    # Cosine similarity: dot(q, c) / (||q|| * ||c||)
    norms = np.linalg.norm(corpus_embeddings, axis=1)
    query_norm = np.linalg.norm(query_embedding)

    # Avoid division by zero
    denom = norms * query_norm
    denom = np.where(denom == 0, 1e-10, denom)

    # Compute actual similarities (using matrix multiplication for faster document-query dot products)
    similarities = corpus_embeddings @ query_embedding / denom

    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [
        {"chunk_idx": int(i), "content": chunks[i], "score": float(similarities[i])}
        for i in top_indices
    ]


# ---------------------------------------------------------------------------
# Unified LLM Client
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    """
    Abstract base class for LLM interaction.

    All backends expose the same `generate()` method so that your RAG code
    does not need to know which backend is being used.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        history: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt:      The user message / prompt.
            history:     Optional conversation history as a list of dicts with
                         keys "role" ("system", "user", "assistant") and "content".
            temperature: Sampling temperature (0 = deterministic, >1 = creative). Optional.
            max_tokens:  Maximum number of tokens to generate. Optional.

        Returns:
            The generated text response as a string.
        """
        ...


class OpenAIClient(LLMClient):
    """Backend for OpenAI-compatible APIs (standard OpenAI, vLLM, Ollama, etc.)."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        from openai import OpenAI

        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model

    def generate(
        self,
        prompt: str,
        history: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages = list(history) if history else []
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort="minimal",
        )
        return response.choices[0].message.content


class AzureOpenAIClient(LLMClient):
    """
    Backend for Azure OpenAI Service (v1 API).

    Uses the standard OpenAI() client with base_url pointing to the Azure
    endpoint's /openai/v1/ path. No api_version parameter is needed.
    See: https://learn.microsoft.com/en-us/azure/foundry/openai/api-version-lifecycle
    """

    def __init__(self, api_key: str, endpoint: str, deployment: str):
        from openai import OpenAI

        # Ensure endpoint ends without trailing slash, then append /openai/v1/
        base_url = (
            endpoint.rstrip("/") + "/openai/v1/"
            if "openai/v1" not in endpoint
            else endpoint
        )
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self._deployment = deployment

    def generate(
        self,
        prompt: str,
        history: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages = list(history) if history else []
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=messages,
            temperature=1,  # NOTE: Azure OpenAI models do not support different temperature values
            max_completion_tokens=max_tokens,
            reasoning_effort="minimal",
        )
        return response.choices[0].message.content


class TransformersClient(LLMClient):
    """
    Backend for local Hugging Face models loaded with the transformers library.

    This loads the model into memory on first instantiation. For small models
    (~1-4B parameters) this works on CPU (slow) or a single consumer GPU.
    For larger models, use quantization or a dedicated GPU.
    """

    def __init__(self, model_name: str, device_map: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading local model: {model_name} (this may take a moment)...")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self._model_name = model_name
        print(f"Model {model_name} loaded successfully.")

    def generate(
        self,
        prompt: str,
        history: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        import torch

        # Build the messages list
        messages = list(history) if history else []
        messages.append({"role": "user", "content": prompt})

        # Use the model's chat template if available, otherwise fall back to
        # a simple concatenation
        if hasattr(self._tokenizer, "apply_chat_template"):
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
            text += "\nASSISTANT:"

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01)  # avoid 0.0 for sampling
                if temperature
                else None,
                do_sample=temperature > 0 if temperature else None,
            )

        # Decode only the newly generated tokens
        generated = self._tokenizer.decode(
            outputs[0][input_length:], skip_special_tokens=True
        )
        return generated.strip()


def create_llm_client() -> LLMClient:
    """
    Factory function that reads environment variables from `.env` and returns
    the appropriate LLMClient backend.

    Priority order:
      1. If AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are set → AzureOpenAIClient.
      2. If OPENAI_API_KEY is set → OpenAIClient (with optional OPENAI_BASE_URL for
         local servers like vLLM / Ollama).
      3. If LOCAL_MODEL_NAME is set → TransformersClient (loads model locally).
      4. Otherwise, raises ValueError.
    """
    load_dotenv()

    # --- Azure OpenAI (v1 API) ---
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if azure_key and azure_endpoint:
        deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-5-nano")
        print(f"→ Using Azure OpenAI v1 API (deployment: {deployment})")
        return AzureOpenAIClient(
            api_key=azure_key,
            endpoint=azure_endpoint,
            deployment=deployment,
        )

    # --- Standard / Local OpenAI-compatible API ---
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        base_url = os.getenv("OPENAI_BASE_URL")
        model = os.getenv("OPENAI_MODEL_NAME", "gpt-5-nano")
        label = f"OpenAI-compatible endpoint ({base_url})" if base_url else "OpenAI API"
        print(f"→ Using {label} (model: {model})")
        return OpenAIClient(api_key=openai_key, model=model, base_url=base_url)

    # --- Local Hugging Face model ---
    local_model = os.getenv("LOCAL_MODEL_NAME")
    if local_model:
        print(f"→ Using local Hugging Face model: {local_model}")
        return TransformersClient(model_name=local_model)

    raise ValueError(
        "No LLM backend configured. Set one of the following in your .env file:\n"
        "  • AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT  (Azure OpenAI)\n"
        "  • OPENAI_API_KEY                                 (OpenAI / local API)\n"
        "  • LOCAL_MODEL_NAME                               (Hugging Face model)"
    )
