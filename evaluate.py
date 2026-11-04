"""
evaluate.py — Compare the Group 2 RAG pipeline against baselines.

Systems compared
----------------
  direct     Direct LLM prompting, no retrieved context (baseline required by the brief).
  naive_rag  "Textbook" RAG: fixed-size 1000-char chunks (200 overlap), semantic top-k,
             simple grounded prompt. This is the "base RAG pipeline" your improvements
             should beat.
  group2     Your pipeline from group_2_rag.py (paragraph-aware chunking, hybrid
             BM25 + semantic retrieval, paragraph-number filtering).

Metrics
-------
  Retrieval (RAG systems only, no LLM needed):
    hit        1 if any chunk sent to the LLM contains the gold evidence passage.
    rr         Reciprocal rank of the first chunk containing the evidence (0 if none).
    ctx_chars  Characters of context sent to the LLM (cost / noise proxy).
  Generation:
    fact_recall  Share of the gold "required facts" found in the answer (string match,
                 numbers normalised so "35 000 000" == "35,000,000").
    abstained    1 if the answer says it cannot answer from the documents.
    correct      factual question -> fact_recall == 1; out-of-scope -> abstained.
    judge_*      Optional LLM-as-judge scores (1-5) for correctness and groundedness.

Usage
-----
  python evaluate.py --retrieval-only              # no API key needed
  python evaluate.py                               # all systems, string metrics
  python evaluate.py --judge                       # + LLM-as-judge
  python evaluate.py --systems direct,group2 --limit 5

Results are written to results/eval_<timestamp>.csv and results/eval_<timestamp>.md.
To add a new variant (e.g. a reranker), write a function with the same signature as
`run_group2` and register it in SYSTEMS.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import time
from datetime import datetime

from dotenv import load_dotenv

from utils import (
    build_bm25_index,
    create_llm_client,
    embed_texts,
    load_embedding_model,
    semantic_search,
)
from group_2_rag import (
    load_data_from_folder,
    paragraph_aware_chunking,
    rag_generate,
    retrieve,
)

DEFAULT_QA = os.path.join("eval", "qa_eu_ai_act.json")
TEMPERATURE = 1.0  # gpt-5-nano only accepts 1.0; lower it for other models to reduce variance
MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------


def norm_ws(text: str) -> str:
    """Lowercase and collapse whitespace (PDF extraction breaks lines mid-sentence)."""
    return re.sub(r"\s+", " ", text).strip().lower()


def norm_answer(text: str) -> str:
    """norm_ws + join digit groups ("35 000 000" / "35,000,000" -> "35000000")."""
    t = norm_ws(text)
    t = re.sub(r"(?<=\d)[ ,.  ](?=\d{3}\b)", "", t)
    t = t.replace("‑", "-").replace("‐", "-")
    return t


ABSTAIN_PATTERNS = [
    r"not (?:in|included in|contained in|present in|mentioned in|covered by|provided in|found in) the (?:provided )?(?:context|documents?|corpus|text)",
    r"(?:context|documents?|text) (?:does|do) not (?:contain|mention|include|provide|say|cover)",
    r"(?:cannot|can't|can not|unable to) (?:answer|determine|find|provide)",
    r"no (?:information|mention)",
    r"is missing",
    r"i don't know|i do not know",
    r"outside (?:the )?scope",
]


def is_abstention(answer: str) -> bool:
    a = norm_ws(answer)
    return any(re.search(p, a) for p in ABSTAIN_PATTERNS)


def fact_recall(answer: str, required_facts: list[list[str]]) -> float:
    if not required_facts:
        return float("nan")
    a = norm_answer(answer)
    hits = sum(any(norm_answer(alt) in a for alt in alts) for alts in required_facts)
    return hits / len(required_facts)


def retrieval_scores(indices: list[int], chunks: list[str], evidence: str | None) -> tuple[float, float]:
    if not evidence:
        return float("nan"), float("nan")
    ev = norm_ws(evidence)
    for rank, idx in enumerate(indices, start=1):
        if ev in norm_ws(chunks[idx]):
            return 1.0, 1.0 / rank
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------


def fixed_size_chunking(raw_docs: list[dict], size: int = 1000, overlap: int = 200):
    chunks, sources = [], []
    for doc in raw_docs:
        text = doc["text"]
        step = size - overlap
        for start in range(0, max(len(text) - overlap, 1), step):
            chunks.append(text[start:start + size])
            sources.append(doc["source"])
    return chunks, sources


def build_indexes(data_dir: str, embed_model) -> dict:
    raw_docs = load_data_from_folder(data_dir)
    if not raw_docs:
        raise SystemExit(f"No PDFs found in {data_dir!r}")

    idx = {}
    for name, chunker in (("naive", fixed_size_chunking), ("group2", paragraph_aware_chunking)):
        chunks, sources = chunker(raw_docs)
        print(f"[{name}] {len(chunks)} chunks, embedding...")
        idx[name] = {
            "chunks": chunks,
            "sources": sources,
            "bm25": build_bm25_index(chunks),
            "emb": embed_texts(chunks, embed_model),
        }
    return idx


# ---------------------------------------------------------------------------
# Systems: each returns (answer | None, retrieved_indices | None, index_name | None)
# ---------------------------------------------------------------------------

NAIVE_SYSTEM_PROMPT = (
    "You are a legal assistant for the EU AI Act. Answer the question using only the "
    "provided context. If the context does not contain the answer, say so."
)


def run_direct(q, llm, idx, embed_model, top_k, retrieval_only):
    if retrieval_only:
        return None, None, None
    answer = llm.generate(
        prompt=q,
        history=[{"role": "system", "content": "You are a helpful assistant."}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return answer, None, None


def run_naive_rag(q, llm, idx, embed_model, top_k, retrieval_only):
    ix = idx["naive"]
    results = semantic_search(q, embed_model, ix["emb"], ix["chunks"], top_k=top_k)
    indices = [r["chunk_idx"] for r in results]
    if retrieval_only:
        return None, indices, "naive"
    context = "\n\n---\n\n".join(ix["chunks"][i] for i in indices)
    answer = llm.generate(
        prompt=f"Context:\n{context}\n\nQuestion: {q}",
        history=[{"role": "system", "content": NAIVE_SYSTEM_PROMPT}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return answer, indices, "naive"


def run_group2(q, llm, idx, embed_model, top_k, retrieval_only):
    ix = idx["group2"]
    if retrieval_only:
        indices = retrieve(q, ix["bm25"], embed_model, ix["emb"], ix["chunks"], top_k=top_k)
        return None, indices, "group2"
    answer, _sources, indices = rag_generate(
        query=q,
        llm_client=llm,
        bm25_index=ix["bm25"],
        embed_model=embed_model,
        corpus_embeddings=ix["emb"],
        chunks=ix["chunks"],
        chunk_sources=ix["sources"],
        top_k=top_k,
        return_indices=True,
    )
    return answer, indices, "group2"


SYSTEMS = {
    "direct": run_direct,
    "naive_rag": run_naive_rag,
    "group2": run_group2,
}

# ---------------------------------------------------------------------------
# LLM-as-judge (optional)
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are grading an answer about the EU AI Act.

Question: {question}
Reference answer: {reference}
Retrieved context given to the system (may be empty): {context}
System answer: {answer}

Score two things from 1 (worst) to 5 (best):
- correctness: agreement with the reference answer (for out-of-scope questions, 5 = clearly declines / says it is not in the documents).
- groundedness: every claim is supported by the retrieved context (if no context was given, judge whether claims are verifiable legal facts rather than hallucinations).

Reply ONLY with JSON: {{"correctness": <int>, "groundedness": <int>, "reason": "<one sentence>"}}"""


def judge(judge_llm, item, answer, context) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=item["question"],
        reference=item["reference_answer"],
        context=context[:6000] if context else "(none)",
        answer=answer,
    )
    raw = judge_llm.generate(prompt=prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        data = {}
    return {
        "judge_correctness": data.get("correctness", float("nan")),
        "judge_groundedness": data.get("groundedness", float("nan")),
        "judge_reason": data.get("reason", raw),
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def mean(values):
    vals = [v for v in values if isinstance(v, (int, float)) and v == v]  # drop NaN
    return statistics.fmean(vals) if vals else float("nan")


def fmt(x):
    return "–" if x != x else f"{x:.2f}"


def summarise(rows, systems, judge_on) -> str:
    cols = ["hit@ctx", "MRR", "ctx_chars", "fact_recall", "correct (factual)", "abstain (out-of-scope)", "false abstain (factual)", "latency_s"]
    if judge_on:
        cols += ["judge_correct", "judge_grounded"]
    lines = ["| system | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for s in systems:
        r = [x for x in rows if x["system"] == s]
        fac = [x for x in r if x["type"] != "out_of_scope"]
        oos = [x for x in r if x["type"] == "out_of_scope"]
        vals = [
            mean(x["hit"] for x in fac),
            mean(x["rr"] for x in fac),
            mean(x["ctx_chars"] for x in r),
            mean(x["fact_recall"] for x in fac),
            mean(x["correct"] for x in fac),
            mean(x["abstained"] for x in oos),
            mean(x["abstained"] for x in fac),
            mean(x["latency_s"] for x in r),
        ]
        if judge_on:
            vals += [mean(x.get("judge_correctness") for x in r), mean(x.get("judge_groundedness") for x in r)]
        cells = [fmt(v) if c != "ctx_chars" else ("–" if v != v else f"{v:.0f}") for c, v in zip(cols, vals)]
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa", default=DEFAULT_QA, help="Path to the gold Q&A JSON file")
    ap.add_argument("--data", default="data", help="Folder with the PDF corpus")
    ap.add_argument("--systems", default=",".join(SYSTEMS), help=f"Comma-separated subset of {list(SYSTEMS)}")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="Only evaluate the first N questions")
    ap.add_argument("--retrieval-only", action="store_true", help="Skip the LLM; score retrieval only")
    ap.add_argument("--judge", action="store_true", help="Also score answers with an LLM judge")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    load_dotenv()
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = set(systems) - set(SYSTEMS)
    if unknown:
        raise SystemExit(f"Unknown systems: {unknown}")

    with open(args.qa, encoding="utf-8") as f:
        qa = json.load(f)[: args.limit]

    embed_model = load_embedding_model()
    idx = build_indexes(args.data, embed_model)
    llm = None if args.retrieval_only else create_llm_client()

    rows = []
    for item in qa:
        for s in systems:
            t0 = time.time()
            try:
                answer, indices, ix_name = SYSTEMS[s](item["question"], llm, idx, embed_model, args.top_k, args.retrieval_only)
            except Exception as e:  # keep going if one call fails
                print(f"  ! {s} failed on {item['id']}: {e}")
                answer, indices, ix_name = f"ERROR: {e}", None, None
            latency = time.time() - t0

            if indices is None and args.retrieval_only:
                continue  # direct prompting has nothing to score without an LLM

            chunks = idx[ix_name]["chunks"] if ix_name else []
            hit, rr = retrieval_scores(indices or [], chunks, item.get("evidence")) if ix_name else (float("nan"),) * 2
            context = "\n\n".join(chunks[i] for i in indices) if indices else ""
            row = {
                "id": item["id"],
                "type": item["type"],
                "system": s,
                "question": item["question"],
                "hit": hit,
                "rr": rr,
                "ctx_chars": len(context) if ix_name else float("nan"),
                "latency_s": latency if not args.retrieval_only else float("nan"),
                "fact_recall": float("nan"),
                "abstained": float("nan"),
                "correct": float("nan"),
                "answer": answer or "",
            }
            if answer is not None:
                row["fact_recall"] = fact_recall(answer, item["required_facts"])
                row["abstained"] = float(is_abstention(answer))
                row["correct"] = float(row["abstained"]) if item["type"] == "out_of_scope" else float(row["fact_recall"] == 1.0)
                if args.judge:
                    row.update(judge(llm, item, answer, context))
            rows.append(row)
            print(f"{item['id']} {s:<10} hit={fmt(hit)} rr={fmt(rr)} recall={fmt(row['fact_recall'])} abstain={fmt(row['abstained'])}")

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.out, f"eval_{stamp}.csv")
    fieldnames = list(dict.fromkeys(k for r in rows for k in r))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    table = summarise(rows, systems, args.judge)
    md_path = os.path.join(args.out, f"eval_{stamp}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Evaluation {stamp}\n\nQ&A set: `{args.qa}` ({len(qa)} questions), top_k={args.top_k}\n\n{table}\n")

    print("\n" + table)
    print(f"\nPer-question results: {csv_path}\nSummary: {md_path}")


if __name__ == "__main__":
    main()
