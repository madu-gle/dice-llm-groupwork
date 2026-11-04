"""
evaluate.py — Compare three systems on 10 EU AI Act questions.

  direct     LLM without any context (baseline)
  naive_rag  fixed-size chunks + semantic search (base RAG)
  group2     our pipeline from main.py

For each question we check whether the passage that answers it (`evidence`) was in the
context sent to the LLM. Answers are saved to evaluation_results.csv and graded by hand
in the `correct` column.

Run:  python evaluate.py
"""

import csv

from dotenv import load_dotenv

from utils import build_bm25_index, create_llm_client, embed_texts, load_embedding_model, semantic_search
from main import load_data_from_folder, paragraph_aware_chunking, rag_generate

TOP_K = 5

# question, evidence (verbatim text from the PDF that answers the question)
QUESTIONS = [
    {
        "question": "What is written in paragraph 155?",
        "evidence": "(155) In order to ensure that providers of high-risk AI systems",
    },
    {
        "question": "What is the maximum fine for engaging in a prohibited AI practice under Article 5?",
        "evidence": "fines of up to EUR 35 000 000",
    },
    {
        "question": "What fine applies for supplying incorrect or misleading information to notified bodies or national competent authorities?",
        "evidence": "fines of up to EUR 7 500 000",
    },
    {
        "question": "From which date do the prohibitions on AI practices apply?",
        "evidence": "Chapters I and II shall apply from 2 February 2025",
    },
    {
        "question": "When does Article 6(1) and the corresponding obligations apply?",
        "evidence": "shall apply from 2 August 2027",
    },
    {
        "question": "Within how many days must a provider report a serious incident caused by a high-risk AI system?",
        "evidence": "15 days after the provider or, where applicable, the deployer, becomes aware of the serious incident",
    },
    {
        "question": "How long must a provider keep the technical documentation of a high-risk AI system?",
        "evidence": "for a period ending 10 years after the high-risk AI system",
    },
    {
        "question": "For how long is a Member State representative on the European Artificial Intelligence Board designated?",
        "evidence": "for a period of three years, renewable once",
    },
    {
        "question": "What must deployers do when their AI system generates a deep fake?",
        "evidence": "constituting a deep fake, shall",
    },
    {
        "question": "Who must ensure a sufficient level of AI literacy, and for whom?",
        "evidence": "sufficient level of AI literacy of",
    },
]


def fixed_size_chunking(raw_docs, size=1000, overlap=200):
    chunks = []
    for doc in raw_docs:
        text = doc["text"]
        for start in range(0, len(text), size - overlap):
            chunks.append(text[start:start + size])
    return chunks


def main():
    load_dotenv()
    llm = create_llm_client()
    embed_model = load_embedding_model()
    raw_docs = load_data_from_folder("data")

    # Naive RAG index
    naive_chunks = fixed_size_chunking(raw_docs)
    naive_emb = embed_texts(naive_chunks, embed_model)

    # Group 2 index
    g2_chunks, g2_sources = paragraph_aware_chunking(raw_docs)
    g2_bm25 = build_bm25_index(g2_chunks)
    g2_emb = embed_texts(g2_chunks, embed_model)

    rows = []
    for q in QUESTIONS:
        question = q["question"]

        # 1. Direct LLM, no context
        direct_answer = llm.generate(prompt=question, temperature=1.0, max_tokens=1024)
        rows.append(("direct", question, None, direct_answer))

        # 2. Naive RAG
        results = semantic_search(question, embed_model, naive_emb, naive_chunks, top_k=TOP_K)
        context = "\n\n".join(r["content"] for r in results)
        naive_answer = llm.generate(
            prompt=f"Context:\n{context}\n\nQuestion: {question}",
            history=[{"role": "system", "content": "Answer the question using only the provided context."}],
            temperature=1.0, max_tokens=1024,
        )
        rows.append(("naive_rag", question, context, naive_answer))

        # 3. Group 2 RAG
        g2_answer, _, indices = rag_generate(
            question, llm, g2_bm25, embed_model, g2_emb, g2_chunks, g2_sources,
            top_k=TOP_K, return_indices=True,
        )
        context = "\n\n".join(g2_chunks[i] for i in indices)
        rows.append(("group2", question, context, g2_answer))

    # Score retrieval; correctness is graded by hand in the CSV
    results = []
    for system, question, context, answer in rows:
        q = next(q for q in QUESTIONS if q["question"] == question)
        retrieved = q["evidence"] in context if context is not None else None
        results.append({"system": system, "question": question, "retrieved": retrieved,
                        "answer": answer, "correct": ""})
        print(f"{system:<10} retrieved={retrieved!s:<5} {question}")

    print("\nsystem      retrieved")
    for system in ["naive_rag", "group2"]:
        r = [x for x in results if x["system"] == system]
        print(f"{system:<10}  {sum(x['retrieved'] for x in r):>4}/{len(r)}")

    with open("evaluation_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("\nAll answers saved to evaluation_results.csv - fill in the 'correct' column by hand")


if __name__ == "__main__":
    main()
