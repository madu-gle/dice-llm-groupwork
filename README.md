# Setup

1. pip install -r requirements.txt
2. create .env file based on example.env and fill out openai key and url

# TODO

1. Documentation, Report
2. More extensive experimentation
3. Clean and structure code
4. Better comparison and metrics against base RAG pipeline

# Evaluation

`evaluate.py` asks 10 EU AI Act questions to three systems:

- `direct`: the LLM without any context (baseline)
- `naive_rag`: fixed 1000-char chunks + semantic search (base RAG)
- `group2`: our pipeline in `main.py`

For each answer it checks whether the passage that answers the question was retrieved, and whether the answer contains an expected keyword.

```bash
python evaluate.py
```

Prints a summary and saves all answers to `evaluation_results.csv`.
