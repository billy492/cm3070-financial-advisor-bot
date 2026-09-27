"""Financial Advisor Bot — an LLM-augmented multi-signal stock advisor.

CM3070 Final Project (Section 4.2, Project Idea 2). Research/demo system only:
no live trading, no real money. The package wires together price loading
(yfinance), classical technical-analysis features, a local-LLM recommender
(Ollama, Llama 3.1 8B), post-hoc confidence calibration, counterfactual
explanations, and a walk-forward backtest evaluation harness.

See ADR-0001/0002 for the locked architecture decisions.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
