# Setup

1. pip install -r requirements.txt
2. create .env file based on example.env and fill out openai key and url

# TODO

1. Documentation, Report
2. More extensive experimentation
3. Clean and structure code
4. Better comparison and metrics against base RAG pipeline

# Evaluation

`evaluate.py` compares three systems on a hand-made gold set (`eval/qa_eu_ai_act.json`,
16 in-corpus questions with the passage that answers them + 2 out-of-scope questions):

| system | what it is |
|---|---|
| `direct` | LLM without any context (baseline required by the brief) |
| `naive_rag` | fixed 1000-char chunks, semantic top-k, simple grounded prompt ("base RAG") |
| `group2` | our pipeline in `group_2_rag.py` |

```bash
python evaluate.py --retrieval-only    # retrieval metrics only, no API key needed
python evaluate.py                     # + answer metrics (fact recall, abstention)
python evaluate.py --judge             # + LLM-as-judge correctness / groundedness
```

Results land in `results/eval_<timestamp>.csv` (per question, incl. answers) and `.md` (summary table).
To test a new improvement, add a `run_<name>` function to `SYSTEMS` in `evaluate.py` and rerun.
Extend the gold set by adding entries to the JSON; `evidence` must be a verbatim snippet from the PDF text.
