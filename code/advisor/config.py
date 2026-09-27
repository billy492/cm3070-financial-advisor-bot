"""Central configuration for the Financial Advisor Bot.

Defines filesystem paths, the train/test temporal split, the local-LLM
endpoint, and the global random seed. All other modules import their constants
from here so that paths and date windows stay consistent across the data,
feature, recommender, calibration, counterfactual and evaluation layers.

Environment overrides:
    OLLAMA_MODEL     overrides :data:`OLLAMA_MODEL`.
    OLLAMA_BASE_URL  overrides :data:`OLLAMA_BASE_URL`.

These let the operator point the recommender at an alternative local model
(e.g. a Hermes fine-tune) or a remote Ollama host without editing source.
No network access happens at import time.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

# --- Filesystem layout -------------------------------------------------------
# PROJECT_ROOT resolves to the ``code/`` directory: this file lives at
# code/advisor/config.py, so two parents up is code/.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CACHE_DIR: Path = DATA_DIR / "cache"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
# Append-only JSONL stores of LLM answers (see ``advisor.recommender.cache``).
# Kept under results/ (not data/) because they are small and make the backtest
# reproducible offline; price data under data/cache/ is git-ignored.
LLM_CACHE_DIR: Path = RESULTS_DIR / "llm_cache"

# --- Temporal split (walk-forward backtest) ----------------------------------
# Training window used for calibration fitting and any in-sample tuning; the
# held-out test window is never seen during model/threshold selection.
TRAIN_START: date = date(2020, 1, 1)
TRAIN_END: date = date(2025, 1, 1)
TEST_START: date = date(2025, 1, 1)
TEST_END: date = date(2026, 6, 1)  # refreshed held-out end

# Earliest date fetched from yfinance on a cache miss. Deliberately well before
# TRAIN_START so the 50-day indicator windows are fully warmed up at the start
# of the training period. (yfinance 1.x no longer honours ``period="max"``
# reliably, so the loader always fetches from an explicit start date.)
HISTORY_START: date = date(2000, 1, 1)

# --- Local LLM endpoint (ADR-0001) -------------------------------------------
# Defaults target a local Ollama server. ``OLLAMA_BASE_URL`` may carry the
# OpenAI-compatible ``/v1`` suffix; the recommender strips it to reach Ollama's
# native ``/api/chat`` endpoint. Both values may be overridden via environment
# variables so experiments can swap models/hosts without code changes.
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Short advisor tags -> Ollama model names. Shared by the Streamlit UI, the
# evaluation CLI (``--advisors llama qwen heuristic``) and result file names
# such as ``results/calibration_<tag>.json``.
ADVISOR_MODELS: dict[str, str] = {
    "llama": "llama3.1:8b",
    "qwen": "qwen3:8b",
}

# --- Reproducibility ---------------------------------------------------------
RANDOM_SEED: int = 42

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "CACHE_DIR",
    "RESULTS_DIR",
    "LLM_CACHE_DIR",
    "TRAIN_START",
    "TRAIN_END",
    "TEST_START",
    "TEST_END",
    "HISTORY_START",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "ADVISOR_MODELS",
    "RANDOM_SEED",
]
