"""LongBench v2 prompt template, answer extraction, and middle-out truncation,
vendored from THUDM/LongBench (v2, repo root).

Upstream: https://github.com/THUDM/LongBench
Pinned commit: 2e00731f8d0bff23dc4325161044d0ed8af94c1e (2025-01)
Vendored verbatim from `pred.py` (extract_answer, prompt-fill, middle-out
truncation in query_llm) + `prompts/0shot.txt`. Scoring (overall + by
difficulty + by length) follows `result.py`.

LongBench v2 (Bai et al., 2024; arXiv:2412.15204) — 503 multiple-choice
questions over long contexts (8k–2M words).
"""
